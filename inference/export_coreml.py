"""Static CoreML export for a trained (optionally palettized) RetinaFace
checkpoint -- fuses the PriorBox anchor grid as a model BUFFER (a genuine
part of the module's own state, computed once for a fixed export image
size) instead of recomputing it via PriorBox(...).generate_anchors() inside
forward() every call, following the same idea as
https://github.com/azeme1/Pytorch_Retinaface_BBFree/blob/master/convert_to_onnx_original.py's
RetinaStaticExportWrapper: "compute the prior grid once, treat it as a
weight" instead of "recompute it every forward call". Box/landmark decoding
(this repo's own utils.box_utils.decode/decode_landmarks) is baked into the
traced graph too, using that fused buffer -- so the exported model takes a
raw (mean-subtracted) image and returns absolute-coordinate boxes/scores/
landmarks for EVERY prior directly; no PriorBox reconstruction needed at
inference time.

Confidence filtering and NMS stay OUTSIDE the graph -- same choice this
project's external/face_detector/model.py already documents for RetinaFace's
OTHER export path (decode-free there; here decode is fused but NMS still
isn't): keeps the traced graph static-shaped and friendly to CoreML
conversion + palettization, since a data-dependent variable-length NMS
output would fight both.

Palettization (apply_palette_selective/select_worth_compressing below) uses
coremltools.optimize.coreml directly -- no dependency on this project's
private training package. num_clusters for this step is read directly off
the cluster-info file (every layer's own table.shape[-1]) rather than passed
on the command line, since that file is now this export's actual source of
truth for what was trained -- see --checkpoint below. The reported
compression ratio is the REAL on-disk .mlpackage size (float32 vs
palettized), not a theoretical weight-only estimate -- this script measures
what a device actually has to store/load.

Takes inference/export_pytorch.py's OWN output as input -- a plain state_dict
(no training-specific keys) plus its matching cluster info, packed into ONE
zip (export_pytorch_batch_hf.py's convention) -- not the raw training
checkpoint: this repo's inference/ scripts never touch this project's
internal training machinery at all, and build from the same
plain-weights-plus-cluster-info source only. --checkpoint takes that one zip
directly; no separate cluster file to manage.

The exported .mlpackage takes a native CoreML image input (uint8, single
image, no batch dim in the public signature) declared RGB -- this project's
fixed external contract for every model it ships, and NOT configurable
(coremltools has no "give me BGR instead" option for ct.ImageType). Every
checkpoint in this repo trains BGR (cv2's convention), so the wrapper is
always built with input_color_order="rgb" here -- telling it the incoming
tensor IS RGB, so it must permute back to BGR internally before running
this checkpoint's own BGR-trained weights. See RetinaStaticExportWrapper in
export_common.py for the permute itself.

Usage:
    python inference/export_coreml.py --network mobilenetv1_0.25 \\
        --checkpoint pytorch_export/mobilenetv1_0.25_c16.zip --image-size 640
    python inference/export_coreml.py --network mobilenetv1_0.25 \\
        --checkpoint pytorch_export/mobilenetv1_0.25_float32.pth   # bare .pth: plain float32, no palettization
    python inference/export_coreml.py --network mobilenetv1 --image-size 640 \\
        --checkpoint-url https://huggingface.co/azemel/retinaface-xs/resolve/main/checkpoints/mobilenetv1/pytorch/mobilenetv1_c12.zip
"""

import argparse
import math
import os
import sys
import tempfile
from pathlib import Path

import coremltools as ct
import coremltools.optimize.coreml as cto
import torch

_RETINA_DIR = Path(__file__).resolve().parents[1]

if str(_RETINA_DIR) not in sys.path:
    sys.path.append(str(_RETINA_DIR))

from config import get_config  # noqa: E402

sys.path.append(str(Path(__file__).resolve().parent))
from export_common import (  # noqa: E402
    RetinaStaticExportWrapper, load_plain_with_clusters, replace_leaky_relu, cluster_count, download_from_url,
)


def compute_precision_for(fmt: str):
    """ct.convert(compute_precision=...) value for a --format choice.

    mlprogram silently defaults to FLOAT16 (tensors/weights, not just
    storage) unless told otherwise -- confirmed to visibly corrupt output
    (speckled color noise) on at least one checkpoint despite the exported
    file being named "*_float32*". neuralnetwork must get None; passing
    anything else for that format is invalid (compute_precision only applies
    to mlprogram)."""
    return ct.precision.FLOAT32 if fmt == "mlprogram" else None


def path_size_mb(path: Path) -> float:
    """Total size in MB. mlprogram's .mlpackage is a directory bundle, not a
    single file, so a plain path.stat() would undercount it."""
    if path.is_dir():
        return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1024 ** 2
    return path.stat().st_size / 1024 ** 2


def compression_ratio(float32_mb: float, optimized_mb: float) -> float:
    """(original float32 size) / (optimized size) -- e.g. 8.0 means 8x smaller."""
    return float32_mb / optimized_mb if optimized_mb else float("inf")


_COREML_SUPPORTED_NBITS = (1, 2, 3, 4, 6, 8)


def _coreml_nbits(num_clusters: int) -> int:
    """coremltools' kmeans palettizer only accepts nbits in
    _COREML_SUPPORTED_NBITS -- notably NOT every width in between (5 and 7
    are missing), hit in practice by num_clusters=128 (needs exactly 7
    bits) -- ValueError: Invalid value of "nbits" (7) for palettization.
    Rounds UP to the smallest supported width that still covers
    num_clusters distinct values, e.g. 128 clusters -> nbits=8 (up to 256
    codebook slots, only 128 actually used by kmeans)."""
    needed = max(1, math.ceil(math.log2(num_clusters)))
    for nbits in _COREML_SUPPORTED_NBITS:
        if nbits >= needed:
            return nbits
    raise ValueError(f"num_clusters={num_clusters} needs more than 8 bits, unsupported by CoreML palettization")


# Index widths coremltools can palettize to (OpPalettizerConfig._VALID_NBITS,
# see _COREML_SUPPORTED_NBITS above) -- whole bits only: there is no 5 or 7, so
# a cluster count that needs 5 or 7 bits is rounded UP (17..32 clusters -> a
# 6-bit table, 65..256 -> 8-bit).
#
# Which of those widths the CoreML GPU path (compute_units CPU_AND_GPU, "MPS"
# on macOS) executes CORRECTLY for a per-output-channel-palettized DENSE
# SPATIAL conv -- measured on a 64->256 3x3 conv (160x160 input) for 14
# cluster counts, relative error vs PyTorch (CPU_ONLY is exact for every
# width):
#     1-bit ~3e-7   OK        2-bit 0.49-0.52  WRONG
#     3-bit ~4e-7   OK        4-bit 0.45-0.59  WRONG
#     6-bit ~5e-7   OK        8-bit 0.17-0.22  WRONG
# 1x1 (pointwise) and depthwise convs are correct at every width tested.
# So a dense spatial conv is only ever given a 1-, 3- or 6-bit table.
_GPU_SAFE_SPATIAL_NBITS = (1, 3, 6)


def is_dense_spatial_conv_weight(shape) -> bool:
    """A conv weight (ndim 4) with a kernel bigger than 1x1 and more than one
    input channel per group -- i.e. NOT pointwise (k=1) and NOT depthwise
    (weight shape (C, 1, k, k))."""
    return len(shape) == 4 and shape[1] > 1 and (shape[2] > 1 or shape[3] > 1)


def gpu_safe_nbits(num_clusters: int) -> int | None:
    """Smallest GPU-safe index width (_GPU_SAFE_SPATIAL_NBITS) that can hold
    num_clusters distinct values -- e.g. 3-4 clusters (would be a wrong 2-bit
    table) -> 3 bits, 9-16 clusters (wrong 4-bit) -> 6 bits. None when even
    6 bits are not enough (> 64 clusters, which would need the wrong 8-bit
    table): such a layer has to stay dense float32."""
    needed = max(1, math.ceil(math.log2(num_clusters)))
    for nbits in _GPU_SAFE_SPATIAL_NBITS:
        if nbits >= needed:
            return nbits
    return None


def _padded_lut_function(nbits: int):
    """lut_function for OpPalettizerConfig(mode="custom"): the channel's own
    distinct values as the table, zero-padded up to 2**nbits entries, so the
    exported width is exactly nbits even when fewer values are needed (what
    "unique" mode can't do -- it always picks the smallest width, which is
    the wrong one on the GPU for 2/4-bit)."""
    import numpy as np

    def lut_function(weight):
        values = np.asarray(weight).reshape(-1)
        distinct = np.unique(values)
        assert len(distinct) <= 2 ** nbits, f"{len(distinct)} distinct values do not fit {nbits} bits"
        lut = np.zeros(2 ** nbits, dtype=values.dtype)
        lut[:len(distinct)] = distinct
        return lut.tolist(), np.searchsorted(distinct, values).tolist()

    return lut_function


def plan_palettization(mlmodel, num_clusters: int, channel_axis: int, weight_threshold: int = 1024) -> dict:
    """{op name: forced nbits or None} for every weight worth palettizing --
    None means "unique" mode picks the width itself (fine for everything
    except dense spatial convs, see _GPU_SAFE_SPATIAL_NBITS); an int forces
    exactly that width via a padded custom table.

    Per-output-channel palettization is only kept where it is a genuine
    on-disk size win, vs. blanket-palettizing everything weight_threshold or
    larger: a palettized tensor doesn't just shrink -- it trades its float32
    storage for a packed index per element PLUS one float32 table per output
    channel (channel_axis group), wrapped in its own constant-lookup op. For
    a tensor with few elements per channel (e.g. a (4,64,1,1) 1x1 conv head:
    64 elements total, 64 channels), the table cost can exceed the raw
    float32 weights it's replacing -- confirmed exactly this on
    mobilenetv1_0.25 k=16 (blanket palettization measured 0.43x: net
    GROWTH, not compression).

    A dense spatial conv is only palettized when a GPU-safe width exists for
    it (gpu_safe_nbits) and every channel really has <= 2**width distinct
    values (a checkpoint that was not clustered per output channel stays
    dense rather than tripping the padded table's assertion).
    """
    import numpy as np

    metadata = cto.get_weights_metadata(mlmodel, weight_threshold=weight_threshold)
    plan = {}
    for name, meta in metadata.items():
        shape = meta.val.shape
        if len(shape) <= channel_axis:
            continue
        forced = None
        index_bits, table_entries = _coreml_nbits(num_clusters), num_clusters
        if is_dense_spatial_conv_weight(shape):
            forced = gpu_safe_nbits(num_clusters)
            if forced is None:
                continue
            per_channel = np.moveaxis(meta.val, channel_axis, 0).reshape(shape[channel_axis], -1)
            if any(len(np.unique(row)) > 2 ** forced for row in per_channel):
                continue
            index_bits, table_entries = forced, 2 ** forced
        n_elements = meta.val.size
        compressed_bytes = math.ceil(n_elements * index_bits / 8) + shape[channel_axis] * table_entries * 4
        if compressed_bytes < n_elements * 4:
            plan[name] = forced
    return plan


def select_worth_compressing(mlmodel, num_clusters: int, channel_axis: int, weight_threshold: int = 1024) -> list[str]:
    """Names of the ops plan_palettization decides to palettize."""
    return list(plan_palettization(mlmodel, num_clusters, channel_axis, weight_threshold))


def apply_palette_selective(mlmodel, num_clusters: int, channel_axis: int, weight_threshold: int = 1024):
    """Per-output-channel palettization, restricted to the ops
    plan_palettization flags as an actual net size win -- everything else is
    left float32 untouched rather than blanket-compressed.

    "unique" mode (the codebook is read directly off the weights' own
    existing distinct values), NOT "kmeans" -- these weights are already
    clustered by this project's own training-time quantization, so a
    codebook already exists per layer; asking coremltools to re-derive one
    via kmeans is both redundant AND, confirmed directly (comparing this
    .mlpackage's real WIDER FACE AP against the un-palettized dense
    conversion of the SAME clustered checkpoint), broken whenever
    num_clusters isn't exactly 2**nbits: cluster_count values in {12, 128}
    (needing nbits=4 nbits=8 but only using 12/128 of the 16/256 available
    codebook slots) silently collapsed real AP to ~0% -- garbage boxes,
    near-uniform ~1.0 scores -- while exact-power-of-2 counts (2, 256)
    matched PyTorch closely. Ie. a coremltools kmeans-palettizer defect
    with unfilled codebook slots, not anything wrong with this graph.
    "unique" mode sidesteps it entirely by reading the codebook off the
    actual distinct values already present instead of re-clustering --
    confirmed to match PyTorch exactly (0.00% AP diff) across every
    cluster_info count tested, unlike kmeans. nbits must NOT be passed for
    "unique" mode -- it's picked up automatically.

    Dense spatial convs are the one exception: "unique" would pick a 2-, 4-
    or 8-bit table for them, all wrong on the CoreML GPU path (see
    _GPU_SAFE_SPATIAL_NBITS), so those get an explicit GPU-safe width via a
    padded custom table instead (or stay dense if none fits)."""
    plan = plan_palettization(mlmodel, num_clusters, channel_axis, weight_threshold)
    config = cto.OptimizationConfig()
    for name, forced in plan.items():
        common = dict(granularity="per_grouped_channel", group_size=1,
                      channel_axis=channel_axis, weight_threshold=weight_threshold)
        if forced is None:
            config.set_op_name(name, cto.OpPalettizerConfig(mode="unique", **common))
        else:
            config.set_op_name(name, cto.OpPalettizerConfig(
                mode="custom", lut_function=_padded_lut_function(forced), **common))
    return cto.palettize_weights(mlmodel, config), list(plan)



def main():
    p = argparse.ArgumentParser()
    p.add_argument("--network", required=True)
    p.add_argument("--checkpoint", default=None,
                    help="inference/export_pytorch.py's output -- EITHER the combined .zip "
                         "(export_pytorch_batch_hf.py's packing, {state_dict, clusters} in one file -- pass this "
                         "and nothing else, no separate clusters file needed) OR a bare .pth (plain state_dict "
                         "only, for a float32 checkpoint with no cluster info to palettize). Mutually exclusive "
                         "with --checkpoint-url.")
    p.add_argument("--checkpoint-url", default=None,
                    help="same as --checkpoint, but downloaded first from this URL, e.g. "
                         "https://huggingface.co/<repo>/resolve/main/checkpoints/<network>/pytorch/"
                         "<network>_<level>.zip. Mutually exclusive with --checkpoint.")
    p.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"),
                    help="bearer token for --checkpoint-url against a private repo -- defaults to the "
                         "HF_TOKEN system variable")
    p.add_argument("--image-size", type=int, default=640)
    p.add_argument("--out-dir", default="coreml_export")
    args = p.parse_args()
    assert bool(args.checkpoint) != bool(args.checkpoint_url), (
        "need exactly one of --checkpoint or --checkpoint-url"
    )
    checkpoint = args.checkpoint or str(download_from_url(args.checkpoint_url, token=args.hf_token))

    cfg = dict(get_config(args.network))
    model, cluster_info = load_plain_with_clusters(cfg, checkpoint)
    num_clusters = cluster_count(cluster_info)

    # ALWAYS "rgb", not a CLI choice: CoreML's ImageType input is
    # unconditionally RGB (ct.ImageType(color_layout=ct.colorlayout.RGB)
    # below), regardless of anything this script's caller might want --
    # there is no way to make CoreML feed this graph BGR pixels, so the
    # wrapper must always permute RGB->BGR internally to match this
    # checkpoint's BGR training convention. Passing "bgr" here was a real,
    # shipped bug: it skipped that permute, so every previously-exported
    # .mlpackage silently ran with red/blue channels swapped -- confirmed
    # by a real WIDER FACE AP crashing to ~0% on CoreML while the exact
    # same checkpoint's PyTorch AP was ~68%. Every .mlpackage built before
    # this fix needs re-exporting.
    wrapper = RetinaStaticExportWrapper(model, cfg, image_size=(args.image_size, args.image_size),
                                        priors_dtype=torch.float16,
                                        input_color_order="rgb").eval()
    replace_leaky_relu(wrapper)  # coremltools 9.0 mlprogram frontend bug, see export_common.py
    sample_x = torch.zeros(1, 3, args.image_size, args.image_size, dtype=torch.float32)

    with torch.no_grad():
        traced = torch.jit.trace(wrapper, sample_x)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Never names the compression method in a shipped artifact's filename,
    # only the cluster count (or "float32").
    tag = f"c{num_clusters}" if num_clusters is not None else "float32"

    # Native CoreML image input -- uint8, single image, declared RGB --
    # this project's fixed convention across every model it ships, and the
    # reason the wrapper above is always built with input_color_order="rgb".
    mlmodel = ct.convert(
        traced,
        convert_to="mlprogram",
        compute_precision=compute_precision_for("mlprogram"),
        inputs=[ct.ImageType(name="input", shape=sample_x.shape, color_layout=ct.colorlayout.RGB)],
        outputs=[ct.TensorType(name="boxes"), ct.TensorType(name="scores"), ct.TensorType(name="landmarks")],
        # per_grouped_channel palettization (apply_palette_selective, below --
        # matches this checkpoint's own per-output-channel compression) is
        # only valid on iOS18+; the default target is older and rejects it
        # at palettize_weights() time with a ValueError, not at convert()
        # time, so this has to be set here even though palettization
        # happens as a separate pass afterward.
        minimum_deployment_target=ct.target.iOS18,
    )

    # Disk size (path_size_mb, a .mlpackage's actual bytes on storage) and
    # memory size (below) are NOT the same number for a palettized model:
    # CoreML's mlprogram runtime dequantizes a packed-indices-plus-small-table
    # op back into a dense float buffer to actually run compute, so the
    # in-RAM footprint at inference time is the same
    # dense-float size regardless of how small the palettized weights made
    # the file on disk -- only storage/download shrinks, not peak memory.
    # Computed straight from the traced PyTorch model's own parameter/buffer
    # count (identical for the float32 and palettized variants, so this one
    # number covers both).
    memory_mb = sum(p.numel() for p in wrapper.parameters()) * 4 / 1024 ** 2
    memory_mb += sum(b.numel() for b in wrapper.buffers() if b.dtype.is_floating_point) * 4 / 1024 ** 2

    if num_clusters is None:
        # No quantization requested -- this float32 .mlpackage IS the
        # deliverable, not just a stepping stone to a palettized one, so it
        # gets saved for real under out_dir.
        fp32_path = out_dir / f"{args.network}_{tag}.mlpackage"
        mlmodel.save(str(fp32_path))
        fp32_disk_mb = path_size_mb(fp32_path)
        print(f"float32 .mlpackage: disk={fp32_disk_mb:.2f} MB, memory (dense, at inference)={memory_mb:.2f} MB "
              f"-> {fp32_path}")
        return

    # Quantized level: the float32 .mlpackage is only ever an intermediate
    # coremltools needs before palettize_weights() -- it's the SAME
    # parameter count as every other level (only the discrete VALUES
    # differ), so saving one per level as a permanent artifact is pure
    # duplication with no information a single reference float32 export
    # doesn't already carry. Measured from a throwaway temp path (still a
    # real on-disk size, just not kept) so the compression ratio below is
    # real, not estimated -- then discarded.
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_fp32_path = Path(tmp_dir) / "fp32_reference.mlpackage"
        mlmodel.save(str(tmp_fp32_path))
        fp32_disk_mb = path_size_mb(tmp_fp32_path)

    palettized, worth_it = apply_palette_selective(mlmodel, num_clusters=num_clusters, channel_axis=0)
    q_path = out_dir / f"{args.network}_{tag}.mlpackage"
    palettized.save(str(q_path))
    q_disk_mb = path_size_mb(q_path)
    ratio = compression_ratio(fp32_disk_mb, q_disk_mb)
    print(f"{args.network}_{tag}.mlpackage ({num_clusters} clusters, output-channel granularity, "
          f"{len(worth_it)}/{len(cluster_info)} layers actually palettized -- the rest stayed float32 as not "
          f"worth it): disk={q_disk_mb:.2f} MB, memory (dense, at inference)={memory_mb:.2f} MB -> {q_path}")
    print(f"REAL on-disk compression ratio (float32 .mlpackage [{fp32_disk_mb:.2f} MB, not kept] -> "
          f"this .mlpackage): {ratio:.2f}x -- note: memory footprint at inference is IDENTICAL between "
          f"the two ({memory_mb:.2f} MB), only on-disk/download size shrinks (see comment above)")


if __name__ == "__main__":
    main()
