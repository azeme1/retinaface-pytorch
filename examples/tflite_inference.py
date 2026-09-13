"""Run face detection with a .tflite from this repo's scripts/export_tflite.py
(or downloaded from Hugging Face). Two things differ from every other
format's examples in this directory -- both confirmed by direct testing,
see scripts/export_tflite.py's module docstring for why:
  1. Input is NHWC ([1, H, W, 3]), not NCHW.
  2. The three output tensors' NAMES don't reliably match their content --
     identify them by their last dimension (4/1/10) instead.

    pip install tensorflow opencv-python numpy
    python examples/tflite_inference.py --model model_int8_dynamic.tflite --image photo.jpg --image-size 640
"""

import argparse

import cv2
import numpy as np
import tensorflow as tf

from _postprocess import postprocess, letterbox_resize, unletterbox, draw_detections


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--image", required=True)
    p.add_argument("--image-size", type=int, default=640, help="must match the size the .onnx was exported with")
    p.add_argument("--conf-threshold", type=float, default=0.5)
    p.add_argument("--nms-threshold", type=float, default=0.4)
    p.add_argument("--out", default="output.jpg")
    args = p.parse_args()

    orig = cv2.imread(args.image, cv2.IMREAD_COLOR)
    canvas, scale = letterbox_resize(orig, args.image_size)  # aspect-preserving, see _postprocess.py

    # BGR, HWC, float32, raw 0-255 pixel values, batch axis added -- NHWC, not NCHW
    x = canvas[None].astype(np.float32)

    interp = tf.lite.Interpreter(model_path=args.model)
    interp.allocate_tensors()
    interp.set_tensor(interp.get_input_details()[0]["index"], x)
    interp.invoke()

    # match by shape, not name -- see module docstring
    outs = {d["shape"][-1]: interp.get_tensor(d["index"]) for d in interp.get_output_details()}
    boxes, scores, landmarks = outs[4], outs[1], outs[10]

    boxes, scores, landmarks = postprocess(boxes, scores, landmarks, args.conf_threshold, args.nms_threshold)
    print(f"{len(boxes)} face(s) detected")

    boxes, landmarks = unletterbox(boxes, landmarks, scale)

    draw_detections(orig, boxes, scores, landmarks)
    cv2.imwrite(args.out, orig)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
