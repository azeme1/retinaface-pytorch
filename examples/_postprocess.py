"""Shared confidence-threshold + NMS postprocessing for every example in
this directory -- every exported format (ONNX/TFLite/CoreML/TFJS/NCNN)
deliberately leaves this OUTSIDE the graph (see scripts/export_common.py's
RetinaStaticExportWrapper docstring), so every example needs it. No
dependency on the original repo -- these examples are meant to run
standalone after downloading a few files from the Hub -- but nms() does
need torch+torchvision (lazily imported, like cv2 below), for
torchvision.ops.nms's own battle-tested CPU/CUDA kernel rather than a
hand-rolled IoU/division-by-zero-prone reimplementation.
"""

import numpy as np


def nms(dets: np.ndarray, thresh: float) -> list[int]:
    import torch
    import torchvision.ops

    boxes = torch.as_tensor(dets[:, :4], dtype=torch.float32)
    scores = torch.as_tensor(dets[:, 4], dtype=torch.float32)
    return torchvision.ops.nms(boxes, scores, thresh).tolist()


def postprocess(boxes: np.ndarray, scores: np.ndarray, landmarks: np.ndarray,
                 conf_threshold: float = 0.5, nms_threshold: float = 0.4, top_k: int = 750):
    """boxes: [N,4], scores: [N,1] or [N], landmarks: [N,10] -- raw decoded
    output straight from any of this repo's exported models. Returns
    (boxes, scores, landmarks) for the detections that survive
    thresholding + NMS, highest score first."""
    scores = scores.reshape(-1)
    keep = scores > conf_threshold
    boxes, scores, landmarks = boxes[keep], scores[keep], landmarks[keep]

    order = scores.argsort()[::-1]
    boxes, scores, landmarks = boxes[order], scores[order], landmarks[order]

    dets = np.hstack([boxes, scores[:, None]]).astype(np.float32, copy=False)
    keep_idx = nms(dets, nms_threshold)[:top_k]
    return boxes[keep_idx], scores[keep_idx], landmarks[keep_idx]


def rescale_to_original(boxes: np.ndarray, landmarks: np.ndarray, image_size: int,
                         native_w: int, native_h: int) -> tuple[np.ndarray, np.ndarray]:
    """Rescales boxes/landmarks from the fixed image_size x image_size
    export canvas back to one image's own native resolution, for a
    NON-aspect-preserving stretch-resize (x and y scaled independently).
    Superseded by letterbox_resize/unletterbox below -- a plain stretch
    here measurably hurts real accuracy (confirmed: 87.33% -> 68.13% mean
    AP on mobilenetv1 c7, Hard nearly halved), so use those instead unless
    something deliberately still wants stretch behavior."""
    if boxes.shape[0] == 0:
        return boxes, landmarks
    sx, sy = native_w / image_size, native_h / image_size
    boxes = boxes.copy()
    boxes[:, [0, 2]] *= sx
    boxes[:, [1, 3]] *= sy
    landmarks = landmarks.copy()
    landmarks[:, 0::2] *= sx
    landmarks[:, 1::2] *= sy
    return boxes, landmarks


# BGR order, matches this repo's own training mean subtraction (rgb_mean in
# RetinaStaticExportWrapper) -- padding with exactly this value means the
# padded region becomes precisely 0 after mean-subtraction, contributing no
# spurious signal to the network.
LETTERBOX_PAD_VALUE_BGR = (104, 117, 123)


def letterbox_resize(img_bgr: np.ndarray, image_size: int, pad_value=LETTERBOX_PAD_VALUE_BGR):
    """Aspect-preserving resize into a fixed image_size x image_size canvas:
    scales the image by ONE uniform factor (so shapes aren't distorted,
    unlike a plain stretch-resize) and places the result at the canvas's
    TOP-LEFT corner, padding only the bottom/right edges. Placing at (0, 0)
    instead of centering is deliberate -- it means unletterbox() below is a
    pure division by scale, no offset subtraction, so there's no separate
    padding math to get wrong when mapping detections back to the original
    image. Returns (canvas, scale) -- scale is what unletterbox() needs.

    Import cv2 lazily here too, same reason as draw_detections below."""
    import cv2

    h, w = img_bgr.shape[:2]
    scale = min(image_size / w, image_size / h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((image_size, image_size, 3), pad_value, dtype=img_bgr.dtype)
    canvas[:new_h, :new_w] = resized
    return canvas, scale


def unletterbox(boxes: np.ndarray, landmarks: np.ndarray, scale: float) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of letterbox_resize's placement: since the resized image
    sits at the canvas's (0, 0) origin with no offset, mapping detections
    back to the original image is a single division by scale."""
    if boxes.shape[0] == 0:
        return boxes, landmarks
    return boxes / scale, landmarks / scale


def draw_detections(image: np.ndarray, boxes: np.ndarray, scores: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    """Draws boxes/landmarks onto a BGR uint8 image (in place, also
    returned). Uses cv2 -- only imported here since it's the one example
    dependency not every deployment target needs."""
    import cv2

    for box, score, lm in zip(boxes, scores, landmarks):
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f"{score:.2f}", (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        pts = lm.astype(int).reshape(5, 2)
        for (px, py) in pts:
            cv2.circle(image, (px, py), 2, (0, 0, 255), -1)
    return image
