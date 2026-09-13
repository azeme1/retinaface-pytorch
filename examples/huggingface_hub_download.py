"""Download exported files from a Hugging Face Hub repo (one scripts/
push_to_hub.py pushed) and run detection with whichever format you want --
reuses this directory's own onnxruntime_inference.py/tflite_inference.py
logic rather than duplicating it.

    pip install huggingface_hub onnxruntime opencv-python numpy
    python examples/huggingface_hub_download.py --repo-id you/retinaface-mobilenetv1-025-quantized \\
        --format onnx --image photo.jpg --image-size 640
"""

import argparse
import runpy
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

_FILENAME_BY_FORMAT = {
    "onnx": "model.onnx",
    "tflite": "tflite/model_int8_dynamic.tflite",
    "coreml": "model.mlpackage",   # a directory -- see the snapshot_download branch below
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo-id", required=True)
    p.add_argument("--format", required=True, choices=["onnx", "tflite", "coreml"])
    p.add_argument("--image", required=True)
    p.add_argument("--image-size", type=int, default=640)
    p.add_argument("--conf-threshold", type=float, default=0.5)
    p.add_argument("--nms-threshold", type=float, default=0.4)
    p.add_argument("--out", default="output.jpg")
    args = p.parse_args()

    if args.format == "coreml":
        # .mlpackage is a directory, not a single file -- pull the whole repo subtree
        local_dir = snapshot_download(args.repo_id, allow_patterns=["model.mlpackage/*"])
        model_path = str(Path(local_dir) / "model.mlpackage")
    else:
        model_path = hf_hub_download(args.repo_id, filename=_FILENAME_BY_FORMAT[args.format])

    print(f"downloaded -> {model_path}")

    script = {"onnx": "onnxruntime_inference.py", "tflite": "tflite_inference.py", "coreml": "coreml_inference.py"}[args.format]
    script_path = Path(__file__).resolve().parent / script

    sys.argv = [
        script, "--model", model_path, "--image", args.image, "--image-size", str(args.image_size),
        "--conf-threshold", str(args.conf_threshold), "--nms-threshold", str(args.nms_threshold), "--out", args.out,
    ]
    runpy.run_path(str(script_path), run_name="__main__")


if __name__ == "__main__":
    main()
