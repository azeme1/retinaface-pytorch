"""Per-output-channel palettization weight compression for an exported ONNX
graph. CoreML has a native op for this (via
coremltools.optimize.coreml.palettize_weights -- see export_coreml.py);
plain ONNX/ONNX Runtime has no equivalent op, so this rebuilds the same
effect with four ops universally supported since early ONNX opsets (Cast,
Add, Reshape, Gather) and readable by any ONNX Runtime version. This gives
a genuinely smaller ON-DISK .onnx file, expanded back to dense weights in
memory when a session loads it -- same "compressed to store, dense to run"
tradeoff CoreML's own palettization makes.

No CLI of its own -- a library function, applied as one step inside
export_onnx.py's own export whenever the source checkpoint has cluster
info (num_clusters is not None; see that script's --no-palettize for
skipping this step deliberately):

    python export_onnx.py --network mobilenetv1 \\
        --checkpoint pytorch_export/mobilenetv1_c7.zip --image-size 640 --out-dir onnx_export

Called directly (e.g. to palettize an already-exported plain .onnx without
re-running the PyTorch export):

    import onnx
    from onnx_palettize import apply_palette_to_onnx

    model = onnx.load("model.onnx")
    model, compressed = apply_palette_to_onnx(model, num_clusters=7)
    print(f"palettized {len(compressed)} weight tensor(s): {compressed}")
    onnx.save(model, "model_palettized.onnx")

NOTE: this compression is ONNX-Runtime-specific and does NOT carry over
into TFLite/TensorFlow.js -- confirmed empirically that onnx2tf's
conversion pipeline constant-folds this whole Cast/Add/Reshape/Gather
chain back into one dense tensor (every input to it is a graph constant,
so from a graph optimizer's perspective it's just precomputable dead
weight, same as TF's own Grappler would do to it downstream). See
export_tflite.py's docstring for the native mechanism used there instead
(TFLite's own int8 dynamic-range quantization).

How it works, per eligible Conv weight tensor of shape (O, I, kH, kW):
a checkpoint produced by per-output-channel weight clustering has at most
`num_clusters` distinct float values within each output channel (already
true of the tensor that reaches this function -- see build_model, which
loads such a checkpoint into its quantized module tree and bakes each
weight back to plain dense Conv2d/Linear still sitting exactly on that
grid). So instead of storing all O*I*kH*kW values as float32:
  - `codebook`: (O, num_clusters) float32 -- the up-to-`num_clusters`
    distinct values actually used by each output channel.
  - `indices`:  (O, I*kH*kW) uint8 -- which codebook entry each original
    element came from. 1 byte instead of 4 is the actual size win.
A tiny per-channel `offset` (channel index * num_clusters) lets a single
flat Gather stand in for a per-channel lookup: Cast(indices, int64) +
offset -> Gather(codebook.flatten(), .) -> Reshape back to (O, I, kH, kW).
That reconstructed tensor feeds the original Conv node exactly as the
initializer did -- inference numerically unchanged, only the on-disk
representation of the weight is smaller.

Skips any weight with more than `num_clusters` distinct values in some
output channel (not actually clustered -- palettizing it here would be
lossy, which this pass never does) or where the reconstruction subgraph
wouldn't net a smaller file (see `_worth_compressing`) -- exactly mirroring
export_coreml.py's own select_worth_compressing heuristic, just against
this scheme's real byte costs instead of coremltools'.
"""
from __future__ import annotations

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

_COMPRESSIBLE_OPS = ("Conv", "Gemm")


def _weight_input_index(node: onnx.NodeProto) -> int | None:
    if node.op_type == "Conv" and len(node.input) > 1:
        return 1
    if node.op_type == "Gemm" and len(node.input) > 1:
        trans_b = next((a.i for a in node.attribute if a.name == "transB"), 0)
        # Gemm's output-channel axis is axis 0 of B only when transB=1 (the
        # convention torch.onnx.export uses for nn.Linear) -- skip the rare
        # transB=0 case rather than palettizing along the wrong axis.
        return 1 if trans_b else None
    return None


def _worth_compressing(out_channels: int, per_channel_elements: int, num_clusters: int) -> bool:
    original_bytes = out_channels * per_channel_elements * 4
    compressed_bytes = (
        out_channels * per_channel_elements * 1       # uint8 indices
        + out_channels * num_clusters * 4             # float32 codebook
        + out_channels * 8                             # int64 per-channel offset
    )
    return compressed_bytes < original_bytes


def _codebook_and_indices(weight: np.ndarray, num_clusters: int) -> tuple[np.ndarray, np.ndarray] | None:
    out_channels = weight.shape[0]
    flat = weight.reshape(out_channels, -1)
    codebook = np.zeros((out_channels, num_clusters), dtype=np.float32)
    indices = np.zeros(flat.shape, dtype=np.uint8)
    for o in range(out_channels):
        uniq, inv = np.unique(flat[o], return_inverse=True)
        if len(uniq) > num_clusters:
            return None  # not actually clustered to num_clusters -- would be lossy, refuse
        codebook[o, :len(uniq)] = uniq
        indices[o] = inv.astype(np.uint8)
    return codebook, indices


def apply_palette_to_onnx(model: onnx.ModelProto, num_clusters: int, weight_threshold: int = 1024) -> tuple[onnx.ModelProto, list[str]]:
    """Palettizes every eligible Conv/Gemm weight initializer in `model`
    in place, skipping any tensor smaller than `weight_threshold` elements
    (not worth a decompression subgraph) or that isn't a net size win.
    Returns (model, [compressed initializer names]) -- the caller reports
    that list so it's obvious which layers actually got compressed."""
    graph = model.graph
    initializer_by_name = {init.name: init for init in graph.initializer}
    decompress_nodes: list[onnx.NodeProto] = []
    new_initializers: list[onnx.TensorProto] = []
    removed_initializers: list[onnx.TensorProto] = []
    compressed = []

    for node in graph.node:
        w_idx = _weight_input_index(node)
        if w_idx is None:
            continue
        w_name = node.input[w_idx]
        init = initializer_by_name.get(w_name)
        if init is None:
            continue

        weight = numpy_helper.to_array(init)
        if weight.size < weight_threshold:
            continue
        out_channels = weight.shape[0]
        per_channel_elements = weight.size // out_channels
        if not _worth_compressing(out_channels, per_channel_elements, num_clusters):
            continue

        result = _codebook_and_indices(weight, num_clusters)
        if result is None:
            continue
        codebook, indices = result

        # Generic node/tensor names ("_table"/"_codes") -- this graph ships
        # to others, and its own internal names shouldn't disclose the
        # compression method any more than the plain PyTorch export's
        # state_dict keys do.
        prefix = w_name.replace("/", "_").replace(".", "_")
        offsets = (np.arange(out_channels, dtype=np.int64) * num_clusters).reshape(out_channels, 1)

        codebook_init = numpy_helper.from_array(codebook.reshape(-1), name=f"{prefix}_table")
        indices_init = numpy_helper.from_array(indices, name=f"{prefix}_codes")
        offset_init = numpy_helper.from_array(offsets, name=f"{prefix}_offset")
        flat_shape_init = numpy_helper.from_array(np.array([-1], dtype=np.int64), name=f"{prefix}_flat_shape")
        orig_shape_init = numpy_helper.from_array(np.array(weight.shape, dtype=np.int64), name=f"{prefix}_orig_shape")

        idx_i64 = f"{prefix}_codes_i64"
        global_idx = f"{prefix}_global_idx"
        global_idx_flat = f"{prefix}_global_idx_flat"
        gathered = f"{prefix}_gathered"
        reconstructed = f"{prefix}_weight"

        decompress_nodes.extend([
            helper.make_node("Cast", [indices_init.name], [idx_i64], to=TensorProto.INT64),
            helper.make_node("Add", [idx_i64, offset_init.name], [global_idx]),
            helper.make_node("Reshape", [global_idx, flat_shape_init.name], [global_idx_flat]),
            helper.make_node("Gather", [codebook_init.name, global_idx_flat], [gathered], axis=0),
            helper.make_node("Reshape", [gathered, orig_shape_init.name], [reconstructed]),
        ])
        new_initializers.extend([codebook_init, indices_init, offset_init, flat_shape_init, orig_shape_init])
        removed_initializers.append(init)
        node.input[w_idx] = reconstructed
        compressed.append(w_name)

    for init in removed_initializers:
        graph.initializer.remove(init)
    graph.initializer.extend(new_initializers)
    # Every new node depends only on initializers (constants), never on
    # another node's activation output, so prepending them all before the
    # graph's existing nodes is always a valid topological order.
    existing_nodes = list(graph.node)
    del graph.node[:]
    graph.node.extend(decompress_nodes + existing_nodes)

    return model, compressed
