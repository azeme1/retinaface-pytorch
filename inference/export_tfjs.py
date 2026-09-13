"""SavedModel -> TensorFlow.js graph model. Converts the SavedModel
export_tflite.py writes (alongside its .tflite files) into a TF.js graph
model: a model.json + one or more sharded weight .bin files, loadable in a
browser or Node.js via @tensorflow/tfjs / tfjs-node (see examples/).

Same reality as TFLite (see export_tflite.py's docstring): the ONNX
palettization pass is constant-folded away by the time this SavedModel
exists, so on-disk compression for this format comes from
tensorflowjs_converter's OWN post-training weight quantization instead --
--quantize_uint8 by default here, a standard, well-supported TF.js feature
(https://www.tensorflow.org/js/guide/conversion#model_conversion). Since
the weights being quantized already sit on a tight, small-cardinality grid,
this loses very little extra precision -- same reasoning export_tflite.py's
dynamic-range int8 default relies on.

Inherits export_tflite.py's SavedModel quirks: NHWC input
([1, H, W, 3], not NCHW) and output tensors whose NAMES land on the wrong
shapes (onnx2tf bug, not this script's) -- identify boxes/scores/landmarks
by their distinctive last dimension (4 / 1 / 10) instead, exactly as
examples/tfjs_inference_node.js does.

Requires the `tensorflowjs` pip package's `tensorflowjs_converter` CLI on
PATH. Install it in a SEPARATE virtualenv from the one used for
export_onnx.py/export_tflite.py: as of writing, tensorflowjs pins an older
tensorflow than onnx2tf needs, so the two don't coexist in one
environment. Since this script only shells out to a CLI against a
SavedModel directory already written to disk, running it from a different
env/machine than export_tflite.py is fine:
    python -m venv /tmp/tfjs-env && /tmp/tfjs-env/bin/pip install tensorflowjs
    /tmp/tfjs-env/bin/python inference/export_tfjs.py --saved-model tflite_export --out-dir tfjs_export

KNOWN BROKEN INSTALL (confirmed at time of writing, tensorflowjs==4.22.0):
a plain `pip install tensorflowjs` into a clean venv fails at IMPORT time,
before this script even runs -- tensorflowjs unconditionally imports
tensorflow_decision_forests (unrelated to model conversion, apparently just
bundled), which pulls in yggdrasil-decision-forests ("ydf"). ydf's
generated protobuf code needs protobuf>=6.31, while tensorflow itself pins
protobuf<6 -- two of tensorflowjs's own transitive dependencies want
mutually exclusive protobuf majors, so no version of protobuf satisfies
both (confirmed: forcing either side just moves the ImportError to the
other). This is an upstream tensorflowjs/tensorflow-decision-forests
packaging bug, nothing this script can route around. If you hit it: check
for a newer tensorflowjs release when you read this (may already be
fixed), or search its GitHub issues for the current workaround -- as of
writing there wasn't a clean one via pip alone.

Usage:
    python inference/export_tfjs.py --saved-model tflite_export --out-dir tfjs_export
"""

import argparse
import shutil
import subprocess
from pathlib import Path


def export_tfjs(saved_model_dir: str, out_dir: str, quantize: str = "uint8") -> Path:
    if shutil.which("tensorflowjs_converter") is None:
        raise RuntimeError(
            "tensorflowjs_converter not found on PATH -- install the `tensorflowjs` pip package "
            "first (preferably in its own virtualenv -- see this module's docstring)."
        )
    if not (Path(saved_model_dir) / "saved_model.pb").exists():
        raise FileNotFoundError(
            f"{saved_model_dir} doesn't look like a SavedModel directory (no saved_model.pb) -- "
            "run export_tflite.py on your .onnx first, it writes one alongside its .tflite files."
        )

    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)

    cmd = [
        "tensorflowjs_converter",
        "--input_format=tf_saved_model",
        "--output_format=tfjs_graph_model",
        "--signature_name=serving_default",
        "--saved_model_tags=serve",
    ]
    if quantize == "uint8":
        cmd.append("--quantize_uint8")
    elif quantize == "float16":
        cmd.append("--quantize_float16")
    elif quantize != "none":
        raise ValueError(f"unknown quantize mode: {quantize!r} (expected uint8, float16, or none)")
    cmd += [str(saved_model_dir), str(out_dir_p)]

    subprocess.run(cmd, check=True)

    size_mb = sum(f.stat().st_size for f in out_dir_p.rglob("*") if f.is_file()) / 1e6
    print(f"TensorFlow.js graph model (quantize={quantize}): {size_mb:.2f} MB -> {out_dir_p}")
    return out_dir_p


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--saved-model", required=True, help="SavedModel directory written by export_tflite.py")
    p.add_argument("--out-dir", default="tfjs_export")
    p.add_argument("--quantize", default="uint8", choices=["uint8", "float16", "none"])
    args = p.parse_args()
    export_tfjs(args.saved_model, args.out_dir, args.quantize)


if __name__ == "__main__":
    main()
