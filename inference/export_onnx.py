"""Static ONNX export for a trained (optionally palettized) RetinaFace
checkpoint -- same RetinaStaticExportWrapper as export_coreml.py (fused
PriorBox buffer, decode baked in, fixed image size; see
inference/export_common.py's docstring), traced with torch.onnx.export
instead of coremltools. Output is boxes/scores/landmarks for every prior;
confidence filtering and NMS stay outside the graph (see
RetinaStaticExportWrapper's docstring for why).

When the checkpoint carries cluster info, the exported graph is palettized
with inference/onnx_palettize.py's Gather-based per-output-channel lookup
table -- the ONNX equivalent of the palettization export_coreml.py already
applies via coremltools, giving a genuinely smaller ON-DISK .onnx file
(confirmed: mobilenetv1_0.25 at 320x320, 16 clusters -- 2.21 MB dense vs
1.17 MB palettized) rather than a dense float32 one whose values merely
happen to repeat; the graph is expanded back to dense weights in memory
at load time, same "compressed to store, dense to run" tradeoff CoreML's
own palettization makes. This compression is ONNX-Runtime-specific --
export_tflite.py's docstring explains why it does NOT survive into
TFLite/TFJS (their conversion pipelines constant-fold it away), and what
those formats use instead.

Takes inference/export_pytorch.py's own output as input -- a plain state_dict
plus its matching cluster-info file -- same convention as export_coreml.py,
not the raw training checkpoint (see that script's docstring for why).
Output is a single zip per level (e.g. mobilenetv1_c16.zip) containing one
generically-named "model.onnx" -- neither the zip name's own directory
context nor the file inside it names the compression method, only the
cluster count (see inference/export_pytorch_batch_hf.py's same convention).

Usage:
    python inference/export_onnx.py --network mobilenetv1_0.25 \\
        --checkpoint pytorch_export/mobilenetv1_0.25_c16.zip --image-size 640
    python inference/export_onnx.py --network mobilenetv1_0.25 \\
        --checkpoint pytorch_export/mobilenetv1_0.25_float32.pth   # bare .pth: plain float32, no palettization
"""

import argparse
import sys
import tempfile
import zipfile
from pathlib import Path

import onnx
import onnxsim
import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import get_config  # noqa: E402

from export_common import RetinaStaticExportWrapper, load_plain_with_clusters, cluster_count  # noqa: E402
from onnx_palettize import apply_palette_to_onnx  # noqa: E402


def path_size_mb(path: str) -> float:
    p = Path(path)
    if p.is_dir():
        return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1e6
    return p.stat().st_size / 1e6


def export_onnx(network: str, checkpoint: str,
                 image_size: int, out_dir: str, opset: int = 17, simplify: bool = True) -> Path:
    cfg = dict(get_config(network))
    model, cluster_info = load_plain_with_clusters(cfg, checkpoint)
    num_clusters = cluster_count(cluster_info)

    wrapper = RetinaStaticExportWrapper(model, cfg, image_size=(image_size, image_size)).eval()
    sample_x = torch.zeros(1, 3, image_size, image_size, dtype=torch.float32)

    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)
    tag = f"c{num_clusters}" if num_clusters is not None else "float32"
    zip_path = out_dir_p / f"{network}_{tag}.zip"

    with tempfile.TemporaryDirectory() as tmp_dir:
        onnx_path = Path(tmp_dir) / "model.onnx"

        torch.onnx.export(
            wrapper, sample_x, str(onnx_path),
            export_params=True, opset_version=opset, do_constant_folding=True,
            input_names=["input"], output_names=["boxes", "scores", "landmarks"],
            # dynamo=True (torch's default since 2.7+) ran into two issues on
            # this graph: a version-converter crash on Resize going to opset 17
            # (this repo's SSH module uses interpolate/Resize) and, after that,
            # an onnxsim topological-sort failure on the graph it produced
            # (a duplicated weight name it introduced, "...weight_1", with no
            # producing node). The older TorchScript-based exporter emits a
            # clean graph for this model with neither problem.
            dynamo=False,
        )

        onnx_model = onnx.load(str(onnx_path))
        if simplify:
            onnx_model, ok = onnxsim.simplify(onnx_model)
            if not ok:
                raise RuntimeError("onnxsim simplification failed to validate")

        if num_clusters is not None:
            onnx_model, compressed = apply_palette_to_onnx(onnx_model, num_clusters=num_clusters)
            print(f"palettized {len(compressed)} weight tensor(s) out of "
                  f"{sum(1 for n in onnx_model.graph.node if n.op_type == 'Conv')} conv layers")

        onnx.checker.check_model(onnx_model)
        onnx.save(onnx_model, str(onnx_path))

        size_mb = path_size_mb(onnx_path)
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(onnx_path, arcname="model.onnx")

    zip_mb = path_size_mb(zip_path)
    print(f"ONNX model ({'palettized' if num_clusters is not None else 'float32'}): "
          f"{size_mb:.2f} MB raw, {zip_mb:.2f} MB zipped -> {zip_path}")
    return zip_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--network", required=True)
    p.add_argument("--checkpoint", required=True,
                    help="inference/export_pytorch.py's output -- EITHER the combined .zip "
                         "(export_pytorch_batch_hf.py's packing, {state_dict, clusters} in one file) OR a bare "
                         ".pth (plain state_dict only, for a float32 checkpoint with no cluster info)")
    p.add_argument("--image-size", type=int, default=640)
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--out-dir", default="onnx_export")
    p.add_argument("--no-simplify", action="store_true", help="skip the onnxsim cleanup pass")
    args = p.parse_args()

    export_onnx(args.network, args.checkpoint,
                args.image_size, args.out_dir, args.opset, simplify=not args.no_simplify)


if __name__ == "__main__":
    main()
