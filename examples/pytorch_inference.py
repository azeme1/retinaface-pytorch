"""Run face detection with pytorch_model.bin (from this repo's scripts/
export_pytorch.py, or downloaded from Hugging Face) -- a plain state_dict
for the ORIGINAL RetinaFace(cfg=cfg) architecture, so this needs the repo
this checkpoint came from (https://github.com/yakhyo/retinaface-pytorch)
on PYTHONPATH, unlike the other examples here. Unlike the exported ONNX/
CoreML/TFLite/TFJS graphs, decode is NOT baked in here -- this is the raw
model, so PriorBox anchor generation + decode happen in this script,
exactly like the repo's own detect.py.

    pip install torch opencv-python numpy
    python examples/pytorch_inference.py --repo-dir /path/to/retinaface-pytorch \\
        --network mobilenetv1_0.25 --weights pytorch_model.bin --image photo.jpg --image-size 640
"""

import argparse
import sys

import cv2
import numpy as np
import torch

from _postprocess import postprocess, letterbox_resize, unletterbox, draw_detections


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo-dir", required=True, help="local clone of yakhyo/retinaface-pytorch")
    p.add_argument("--network", required=True)
    p.add_argument("--weights", required=True)
    p.add_argument("--image", required=True)
    p.add_argument("--image-size", type=int, default=640)
    p.add_argument("--conf-threshold", type=float, default=0.5)
    p.add_argument("--nms-threshold", type=float, default=0.4)
    p.add_argument("--out", default="output.jpg")
    args = p.parse_args()

    sys.path.insert(0, args.repo_dir)
    from config import get_config
    from models import RetinaFace
    from layers import PriorBox
    from utils.box_utils import decode, decode_landmarks

    cfg = get_config(args.network)
    model = RetinaFace(cfg=cfg)
    model.load_state_dict(torch.load(args.weights, map_location="cpu", weights_only=True))
    model.eval()

    orig = cv2.imread(args.image, cv2.IMREAD_COLOR)
    canvas, letterbox_scale = letterbox_resize(orig, args.image_size)  # aspect-preserving, see _postprocess.py
    canvas = canvas.astype(np.float32)
    canvas -= (104, 117, 123)   # bgr mean, matches this repo's own train.py
    x = torch.from_numpy(canvas.transpose(2, 0, 1)).unsqueeze(0)

    with torch.no_grad():
        loc, conf, landmarks = model(x)
    loc, conf, landmarks = loc.squeeze(0), conf.squeeze(0), landmarks.squeeze(0)

    priors = PriorBox(cfg, image_size=(args.image_size, args.image_size)).generate_anchors()
    canvas_scale = args.image_size  # normalized (0-1) decode output -> canvas pixel coords
    boxes = (decode(loc, priors, cfg["variance"]) * canvas_scale).numpy()
    landmarks = (decode_landmarks(landmarks, priors, cfg["variance"]) * canvas_scale).numpy()
    scores = conf[:, 1:2].numpy()

    boxes, scores, landmarks = postprocess(boxes, scores, landmarks, args.conf_threshold, args.nms_threshold)
    print(f"{len(boxes)} face(s) detected")

    boxes, landmarks = unletterbox(boxes, landmarks, letterbox_scale)

    draw_detections(orig, boxes, scores, landmarks)
    cv2.imwrite(args.out, orig)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
