"""Run face detection with model.onnx (from this repo's inference/export_onnx.py,
or downloaded from the Hugging Face repo it was pushed to). Works
identically whether the file is a plain dense export or a palettized one
(inference/onnx_palettize.py) -- the decompression, if any, is baked into
the graph, invisible from here.

    pip install onnxruntime opencv-python numpy
    python examples/onnxruntime_inference.py --model model.onnx --image photo.jpg --image-size 640
"""

import argparse

import cv2
import numpy as np
import onnxruntime as ort

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

    # BGR, HWC -> CHW, float32, raw 0-255 pixel values (mean-subtraction happens inside the graph)
    x = canvas.transpose(2, 0, 1)[None].astype(np.float32)

    sess = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    boxes, scores, landmarks = sess.run(None, {"input": x})

    boxes, scores, landmarks = postprocess(boxes, scores, landmarks, args.conf_threshold, args.nms_threshold)
    print(f"{len(boxes)} face(s) detected")

    boxes, landmarks = unletterbox(boxes, landmarks, scale)

    draw_detections(orig, boxes, scores, landmarks)
    cv2.imwrite(args.out, orig)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
