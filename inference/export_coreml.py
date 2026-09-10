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
fixed external contract for every model it ships, regardless of what
channel order any given checkpoint was actually trained in (this repo
always trains BGR, cv2's convention). --input-color-order tells the wrapped
model which permute (if any) to apply internally to get from that RGB input
back to what its own weights expect -- see RetinaStaticExportWrapper in
export_common.py.

Usage:
    python inference/export_coreml.py --network mobilenetv1_0.25 \\
        --checkpoint pytorch_export/mobilenetv1_0.25_c16.zip --image-size 640
    python inference/export_coreml.py --network mobilenetv1_0.25 \\
        --checkpoint pytorch_export/mobilenetv1_0.25_float32.pth   # bare .pth: plain float32, no palettization
    python inference/export_coreml.py --network mobilenetv1 --image-size 640 \\
        --checkpoint-url https://huggingface.co/azemel/retinaface-xs/resolve/main/results/mobilenetv1/pytorch/mobilenetv1_c12.zip
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


def select_worth_compressing(mlmodel, num_clusters: int, channel_axis: int, weight_threshold: int = 1024) -> list[str]:
    """Names of ops where per-output-channel palettization is a genuine
    on-disk size win, vs. blanket-palettizing everything weight_threshold
    or larger.

    Needed because a palettized tensor doesn't just shrink -- it trades its
    float32 storage for TWO things: a packed index per element (ceil(log2(
    num_clusters)) bits) PLUS one float32 table per output channel (channel_axis
    group) of num_clusters entries, wrapped in its own constant-lookup
    op. For a tensor with few elements per channel (e.g. a (4,64,1,1) 1x1
    conv head: 64 elements total, 64 channels), the per-channel table cost can
    exceed the raw float32 weights it's replacing -- confirmed exactly this
    on mobilenetv1_0.25 k=16 (blanket palettization measured 0.43x: net
    GROWTH, not compression). Filtering to only ops that pass this estimate
    is what actually produces a smaller .mlpackage.
    """
    metadata = cto.get_weights_metadata(mlmodel, weight_threshold=weight_threshold)
    index_bits = max(1, math.ceil(math.log2(num_clusters)))
    worth_it = []
    for name, meta in metadata.items():
        shape = meta.val.shape
        if len(shape) <= channel_axis:
            continue
        n_elements = meta.val.size
        n_groups = shape[channel_axis]
        compressed_bytes = math.ceil(n_elements * index_bits / 8) + n_groups * num_clusters * 4
        float32_bytes = n_elements * 4
        if compressed_bytes < float32_bytes:
            worth_it.append(name)
    return worth_it


def apply_palette_selective(mlmodel, num_clusters: int, channel_axis: int, weight_threshold: int = 1024):
    """Per-output-channel kmeans palettization, restricted to the ops
    select_worth_compressing flags as an actual net size win -- everything
    else is left float32 untouched rather than blanket-compressed."""
    worth_it = select_worth_compressing(mlmodel, num_clusters, channel_axis, weight_threshold)
    nbits = max(1, math.ceil(math.log2(num_clusters)))
    palettizer = cto.OpPalettizerConfig(
        mode="kmeans", nbits=nbits, granularity="per_grouped_channel",
        group_size=1, channel_axis=channel_axis, weight_threshold=weight_threshold,
    )
    config = cto.OptimizationConfig()
    for name in worth_it:
        config.set_op_name(name, palettizer)
    return cto.palettize_weights(mlmodel, config), worth_it


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
                         "https://huggingface.co/<repo>/resolve/main/results/<network>/pytorch/"
                         "<network>_<level>.zip. Mutually exclusive with --checkpoint.")
    p.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"),
                    help="bearer token for --checkpoint-url against a private repo -- defaults to the "
                         "HF_TOKEN system variable")
    p.add_argument("--input-color-order", default="bgr", choices=["bgr", "rgb"],
                    help="channel order this checkpoint's weights were actually trained in (this repo: always "
                         "bgr, cv2's convention) -- NOT the exported model's own public input format, which is "
                         "always a native RGB CoreML image regardless of this flag; see RetinaStaticExportWrapper")
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

    wrapper = RetinaStaticExportWrapper(model, cfg, image_size=(args.image_size, args.image_size),
                                        priors_dtype=torch.float16,
                                        input_color_order=args.input_color_order).eval()
    replace_leaky_relu(wrapper)  # coremltools 9.0 mlprogram frontend bug, see export_common.py
    sample_x = torch.zeros(1, 3, args.image_size, args.image_size, dtype=torch.float32)

    with torch.no_grad():
        traced = torch.jit.trace(wrapper, sample_x)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Never names the compression method in a shipped artifact's filename,
    # only the cluster count (or "float32").
    tag = f"c{num_clusters}" if num_clusters is not None else "float32"

    # Native CoreML image input -- uint8, single image, declared RGB
    # regardless of --input-color-order (that flag only controls the
    # in-graph permute back to this checkpoint's own training order; the
    # exported model's public input contract is always RGB, this project's
    # fixed convention across every model it ships).
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
