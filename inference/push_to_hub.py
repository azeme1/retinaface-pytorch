"""Stages the export_*.py outputs into one clean, generically-named
directory and pushes it (plus a generated model card) to a Hugging Face
Hub model repo via huggingface_hub. Doesn't run any export itself -- each
format has its own environment quirks (CoreML segfaults if imported
alongside a mismatched tensorflow/torch, TFJS needs a separate venv, see
those scripts' own docstrings), so this only packages already-produced
files. Run whichever of export_pytorch.py / export_onnx.py /
export_tflite.py / export_coreml.py / export_tfjs.py apply first, then
point this script at their outputs.

Deliberately generic file names (model.onnx, not mobilenetv1_0.25_c16.
onnx) and a model card that describes what a file IS and how to load it,
never how the weight values were produced -- no cluster counts, method
names, or script references. If a per-output-channel lookup table is what
makes an ONNX/CoreML file smaller than a dense one, the card says exactly
that (a user loading the file benefits from knowing it decodes itself,
same as CoreML's own palettization needs no special runtime) -- but not
what training procedure put the weights on that grid in the first place.

Usage:
    python inference/push_to_hub.py --repo-id you/retinaface-mobilenetv1-025-quantized \\
        --pytorch pytorch_export/mobilenetv1_0.25_c16.pth \\
        --onnx onnx_export/mobilenetv1_0.25_c16.onnx \\
        --tflite-dir tflite_export \\
        --coreml coreml_export/mobilenetv1_0.25_c16.mlpackage \\
        --tfjs-dir tfjs_export \\
        --network mobilenetv1_0.25 --image-size 640 --private
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from huggingface_hub import HfApi

_MODEL_CARD_TEMPLATE = """---
license: mit
tags:
- object-detection
- face-detection
- retinaface
- pytorch
- onnx
- tflite
- coreml
- tensorflow.js
- quantized
---

# {title}

A quantized {network} RetinaFace face detector: {size_note}face detection
with bounding boxes, confidence scores, and 5-point facial landmarks.

Every file below decodes raw network output into absolute-coordinate
boxes/landmarks internally where the format supports it -- callers only
need to do confidence thresholding and non-max suppression on top (kept
outside the graph everywhere so the exported shapes stay fixed-size).
Output order/shape (except where a format's own docs say otherwise below):
`boxes [N, 4]` (x1, y1, x2, y2, pixels), `scores [N, 1]`, `landmarks
[N, 10]` (5 (x, y) pairs, pixels), N = the total prior/anchor count for a
{image_size}x{image_size} input.

## Files
{file_list}

## Usage
{usage_sections}
## License
MIT, matching the base architecture this checkpoint fine-tunes
(https://github.com/yakhyo/retinaface-pytorch).
"""


def _human_size(path: Path) -> str:
    if path.is_dir():
        total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    else:
        total = path.stat().st_size
    return f"{total / 1e6:.2f} MB"


def stage(out_dir: str, network: str, pytorch_path: str | None, onnx_path: str | None,
          tflite_dir: str | None, coreml_path: str | None, tfjs_dir: str | None) -> tuple[Path, list[str]]:
    """Copies each provided artifact into out_dir under a clean, generic
    name. Returns (staged_dir, markdown bullet list of what's included)."""
    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)
    file_list = []

    if pytorch_path:
        dst = out_dir_p / "pytorch_model.bin"
        shutil.copy2(pytorch_path, dst)
        file_list.append(f"- `pytorch_model.bin` ({_human_size(dst)}) -- plain PyTorch `state_dict`")

    if onnx_path:
        dst = out_dir_p / "model.onnx"
        shutil.copy2(onnx_path, dst)
        file_list.append(f"- `model.onnx` ({_human_size(dst)}) -- ONNX Runtime")

    if tflite_dir:
        tflite_dst = out_dir_p / "tflite"
        tflite_dst.mkdir(exist_ok=True)
        rename_map = {
            "float32": "model_float32.tflite",
            "float16": "model_float16.tflite",
            "dynamic_range_quant": "model_int8_dynamic.tflite",
        }
        for f in Path(tflite_dir).glob("*.tflite"):
            new_name = next((v for k, v in rename_map.items() if k in f.name), f.name)
            dst = tflite_dst / new_name
            shutil.copy2(f, dst)
            file_list.append(f"- `tflite/{new_name}` ({_human_size(dst)}) -- TFLite")

    if coreml_path:
        dst = out_dir_p / "model.mlpackage"
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(coreml_path, dst)
        file_list.append(f"- `model.mlpackage` ({_human_size(dst)}) -- CoreML (iOS/macOS)")

    if tfjs_dir:
        tfjs_dst = out_dir_p / "tfjs"
        if tfjs_dst.exists():
            shutil.rmtree(tfjs_dst)
        shutil.copytree(tfjs_dir, tfjs_dst)
        file_list.append(f"- `tfjs/model.json` + shards ({_human_size(tfjs_dst)}) -- TensorFlow.js")

    return out_dir_p, file_list


_USAGE_PYTORCH = """### PyTorch
```python
import torch
from models import RetinaFace   # from https://github.com/yakhyo/retinaface-pytorch
from config import get_config

cfg = get_config("{network}")
model = RetinaFace(cfg=cfg)
model.load_state_dict(torch.load("pytorch_model.bin", map_location="cpu", weights_only=True))
model.eval()
```
This is the same `RetinaFace` class and config this repo's own float32
checkpoints load into -- no custom modules required.
"""

_USAGE_ONNX = """### ONNX Runtime
```python
import onnxruntime as ort
import numpy as np

sess = ort.InferenceSession("model.onnx", providers=["CPUExecutionProvider"])
# image: BGR, HWC, float32, {image_size}x{image_size}, NOT normalized (mean-subtraction happens inside the graph)
x = image.transpose(2, 0, 1)[None].astype(np.float32)   # -> [1, 3, {image_size}, {image_size}]
boxes, scores, landmarks = sess.run(None, {{"input": x}})
```
{onnx_size_note}
"""

_USAGE_TFLITE = """### TFLite
```python
import tensorflow as tf
import numpy as np

interp = tf.lite.Interpreter(model_path="tflite/model_int8_dynamic.tflite")
interp.allocate_tensors()
inp = interp.get_input_details()[0]
# NOTE: NHWC input here, unlike every other format's NCHW -- see caveat below
x = image[None].astype(np.float32)   # image: BGR, HWC, {image_size}x{image_size}, raw pixel values
interp.set_tensor(inp["index"], x)
interp.invoke()
# NOTE: match outputs by their last dimension (4/1/10), not by name -- see caveat below
outs = {{d["shape"][-1]: interp.get_tensor(d["index"]) for d in interp.get_output_details()}}
boxes, scores, landmarks = outs[4], outs[1], outs[10]
```
**Caveats specific to this file:** input is NHWC (`[1, {image_size}, {image_size}, 3]`), and the three
output tensors' NAMES don't reliably line up with which is which -- always identify them by
their last-dimension size (4 = boxes, 1 = scores, 10 = landmarks) as shown above.
"""

_USAGE_COREML = """### CoreML
```python
import coremltools as ct
import numpy as np

model = ct.models.MLModel("model.mlpackage")
# image: BGR, HWC, float32, {image_size}x{image_size}, raw pixel values
x = image.transpose(2, 0, 1)[None].astype(np.float32)
out = model.predict({{"input": x}})
boxes, scores, landmarks = out["boxes"], out["scores"], out["landmarks"]
```
"""

_USAGE_TFJS = """### TensorFlow.js (Node.js, `@tensorflow/tfjs-node`)
```js
const tf = require('@tensorflow/tfjs-node');

const model = await tf.loadGraphModel('file://./tfjs/model.json');
// image: BGR, HWC, float32, {image_size}x{image_size}, raw pixel values, NHWC like TFLite above
const input = tf.tensor(imageData, [1, {image_size}, {image_size}, 3], 'float32');
const outputs = model.execute(input);   // array of 3 tensors -- match by .shape[-1] same as TFLite (4/1/10)
```
See `examples/tfjs_inference_node.js` for a complete, runnable version (image decode included).
"""


def write_model_card(out_dir: Path, network: str, image_size: int, file_list: list[str], onnx_palettized: bool,
                      has_pytorch: bool, has_onnx: bool, has_tflite: bool, has_coreml: bool, has_tfjs: bool):
    title = f"{network} RetinaFace (quantized)"
    onnx_size_note = (
        "This file stores its convolution weights with a per-output-channel lookup table "
        "(a small index per weight plus a small float table per channel) instead of one float32 "
        "per weight -- decoded back into normal weights automatically the moment ONNX Runtime "
        "loads the graph, so the code above needs no changes either way."
        if onnx_palettized else
        "Plain dense float32 weights."
    )

    sections = []
    if has_pytorch:
        sections.append(_USAGE_PYTORCH.format(network=network))
    if has_onnx:
        sections.append(_USAGE_ONNX.format(image_size=image_size, onnx_size_note=onnx_size_note))
    if has_tflite:
        sections.append(_USAGE_TFLITE.format(image_size=image_size))
    if has_coreml:
        sections.append(_USAGE_COREML.format(image_size=image_size))
    if has_tfjs:
        sections.append(_USAGE_TFJS.format(image_size=image_size))
    usage_sections = "\n" + "\n".join(sections) if sections else "\n(no exported formats staged)\n"

    card = _MODEL_CARD_TEMPLATE.format(
        title=title, network=network, size_note="lightweight ",
        image_size=image_size, file_list="\n".join(file_list) if file_list else "- (none staged)",
        usage_sections=usage_sections,
    )
    (out_dir / "README.md").write_text(card)
    print(f"Model card -> {out_dir / 'README.md'}")


def push(out_dir: Path, repo_id: str, private: bool):
    api = HfApi()
    api.create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
    api.upload_folder(folder_path=str(out_dir), repo_id=repo_id, repo_type="model")
    print(f"Pushed -> https://huggingface.co/{repo_id}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo-id", required=True, help="e.g. your-username/retinaface-mobilenetv1-025-quantized")
    p.add_argument("--network", required=True)
    p.add_argument("--image-size", type=int, default=640)
    p.add_argument("--pytorch", default=None, help="path from export_pytorch.py")
    p.add_argument("--onnx", default=None, help="path from export_onnx.py")
    p.add_argument("--onnx-palettized", action="store_true",
                    help="set if --onnx was exported with --num-clusters (affects the model card's wording)")
    p.add_argument("--tflite-dir", default=None, help="--out-dir from export_tflite.py")
    p.add_argument("--coreml", default=None, help=".mlpackage path from export_coreml.py")
    p.add_argument("--tfjs-dir", default=None, help="--out-dir from export_tfjs.py")
    p.add_argument("--out-dir", default="hf_export", help="local staging directory")
    p.add_argument("--private", action="store_true")
    p.add_argument("--no-push", action="store_true", help="stage and write the model card only, skip the actual upload")
    args = p.parse_args()

    out_dir, file_list = stage(args.out_dir, args.network, args.pytorch, args.onnx,
                                args.tflite_dir, args.coreml, args.tfjs_dir)
    write_model_card(out_dir, args.network, args.image_size, file_list, args.onnx_palettized,
                      has_pytorch=bool(args.pytorch), has_onnx=bool(args.onnx),
                      has_tflite=bool(args.tflite_dir), has_coreml=bool(args.coreml), has_tfjs=bool(args.tfjs_dir))

    if args.no_push:
        print(f"Staged at {out_dir} -- skipping upload (--no-push)")
    else:
        push(out_dir, args.repo_id, args.private)


if __name__ == "__main__":
    main()
