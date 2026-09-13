"""NCNN export for a trained (optionally palettized) RetinaFace
checkpoint, via pnnx (https://github.com/pnnx/pnnx, the ncnn project's own
PyTorch-to-ncnn converter -- installed as the `pnnx` pip package, which
bundles a prebuilt binary; no ncnn build needed to run this script).

READ BEFORE USING -- unlike every other export in this directory (ONNX,
CoreML, TFLite, TFJS, all confirmed numerically correct against PyTorch to
float precision), this one could NOT be verified correct, after real
effort spent trying. Two separate, reproducible problems, confirmed by
direct testing on this repo's own mobilenetv1_0.25 checkpoint:

1. Exporting RetinaStaticExportWrapper (the fused-priors, decode-baked-in
   wrapper every other backend uses -- see export_common.py) converts
   without error, but its extra buffer/broadcast ops (priors, bbox_scale,
   landmark_scale) trigger pnnx warnings ("binaryop broadcast across batch
   axis... not supported", "unbind along batch axis... not supported")
   that turn out not to be harmless: the resulting boxes/scores outputs
   are numerically wrong (not close to PyTorch), and extracting the
   landmarks output outright segfaults ncnn's own native runtime --
   reproducible regardless of how the landmark decode math is written
   (tried two different, mathematically-equivalent 2D formulations; both
   crash the same way), and independent of fp16 vs fp32 weight storage.

2. Exporting the plain model instead (no wrapper -- raw loc/conf/landmarks
   output, decode left to be done separately, same as this repo's own
   detect.py/evaluate_widerface.py already do) avoids the crash, but the
   per-scale permute+reshape+concat pattern in ClassHead/BboxHead/
   LandmarkHead (flattening each FPN level's (B, num_anchors*C, H, W) map
   before concatenating across scales) comes out of ncnn in a DIFFERENT
   anchor order than this repo's own PriorBox generates -- confirmed the
   value distributions match closely (same min/max/mean, sorted arrays
   nearly equal) but per-row values don't line up, meaning it's an
   ordering mismatch, not corrupted math. Reconciling that order would
   mean reverse-engineering ncnn's actual per-level flattening from the
   pnnx-generated ncnn_debug.py and rebuilding a matching PriorBox --
   not attempted here.

This script exports the PLAIN model (path 2 -- no crash, at least
something usable to start from) and prints the same warning. If NCNN
deployment is a hard requirement: read ncnn_debug.py (written next to the
.param/.bin, see below) to work out the actual anchor order, try ncnn's
older onnx2ncnn converter (a different code path, from export_onnx.py's
output, that may lower this pattern differently), or use one of this
directory's other four export formats, which don't have this problem.

Usage:
    python inference/export_ncnn.py --network mobilenetv1_0.25 \\
        --checkpoint pytorch_export/mobilenetv1_0.25_c16.zip --image-size 640 --out-dir ncnn_export
    python inference/export_ncnn.py --network mobilenetv1_0.25 \\
        --checkpoint pytorch_export/mobilenetv1_0.25_float32.pth   # bare .pth: plain float32
"""

import argparse
import sys
from pathlib import Path

import pnnx
import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import get_config  # noqa: E402

from export_common import load_plain_with_clusters, cluster_count  # noqa: E402

_WARNING = (
    "\nWARNING: this export's anchor order has NOT been verified to match this repo's PriorBox "
    "output -- read this script's module docstring before relying on it for real detections.\n"
)


def export_ncnn(network: str, checkpoint: str, image_size: int, out_dir: str, fp16: bool = True) -> tuple[Path, Path]:
    cfg = dict(get_config(network))
    model, cluster_info = load_plain_with_clusters(cfg, checkpoint)
    num_clusters = cluster_count(cluster_info)  # informational only -- this export never palettizes
    sample_x = torch.zeros(1, 3, image_size, image_size, dtype=torch.float32)

    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)
    tag = f"c{num_clusters}" if num_clusters is not None else "float32"
    param_path = out_dir_p / f"{network}_{tag}.ncnn.param"
    bin_path = out_dir_p / f"{network}_{tag}.ncnn.bin"

    with torch.no_grad():
        traced = torch.jit.trace(model, sample_x)
    ts_path = out_dir_p / f"{network}_{tag}_traced.pt"
    traced.save(str(ts_path))

    pnnx.convert(
        str(ts_path),
        input_shapes=[[1, 3, image_size, image_size]],
        input_types=["f32"],
        ncnnparam=str(param_path),
        ncnnbin=str(bin_path),
        ncnnpy=str(out_dir_p / f"{network}_{tag}_ncnn_debug.py"),  # human-readable dump -- see module docstring
        fp16=fp16,
    )
    ts_path.unlink()

    total_mb = (param_path.stat().st_size + bin_path.stat().st_size) / 1e6
    print(f"NCNN model ({'fp16' if fp16 else 'fp32'} weights, RAW loc/conf/landmarks -- no decode baked in): "
          f"{total_mb:.2f} MB -> {param_path}, {bin_path}")
    print(_WARNING)
    return param_path, bin_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--network", required=True)
    p.add_argument("--checkpoint", required=True,
                    help="inference/export_pytorch.py's output -- EITHER the combined .zip "
                         "(export_pytorch_batch_hf.py's packing, {state_dict, clusters} in one file) OR a bare "
                         ".pth (plain state_dict only, for a float32 checkpoint with no cluster info)")
    p.add_argument("--image-size", type=int, default=640)
    p.add_argument("--out-dir", default="ncnn_export")
    p.add_argument("--no-fp16", action="store_true", help="store weights as float32 instead of pnnx's fp16 default")
    args = p.parse_args()

    export_ncnn(args.network, args.checkpoint, args.image_size, args.out_dir, fp16=not args.no_fp16)


if __name__ == "__main__":
    main()
