"""Evaluates Google's BlazeFace (front-camera, 128x128 input) on the real
WIDER FACE validation set, reusing this repo's own ground-truth/AP-scoring
code (widerface_eval.run_widerface_evaluation) so the numbers are directly
comparable to every RetinaFace backbone's full-val table.

BlazeFace itself is NOT trained here -- there is no official PyTorch
training code or checkpoint from Google (it ships only as a MediaPipe TFLite
graph). This uses the well-known open-source PyTorch port/weights from
https://github.com/hollance/BlazeFace-PyTorch (cloned into
external/blazeface-pytorch), which already carries the TFLite weights
converted into an equivalent nn.Module -- no training/conversion step of our
own is needed to get a real, working detector.

BlazeFace's model/eval interface has nothing in common with RetinaFace's
(fixed 128x128 input, normalized ymin/xmin/ymax/xmax + 6 keypoints + a single
sigmoid score, already NMS'd inside predict_on_image) -- reusing
generate_widerface_predictions (built entirely around RetinaFace's
loc/conf/landmarks + PriorBox decode) isn't possible, so this script writes
WIDER FACE prediction .txt files (the same "name / count / x y w h score"
format, same event/image directory layout) directly, then calls the same
run_widerface_evaluation used for every other backbone in this sweep.

Usage:
    python eval_blazeface_widerface.py --device cpu
"""
import argparse
import sys
from pathlib import Path

RETINA_DIR = Path(__file__).resolve().parent
BLAZEFACE_DIR = RETINA_DIR.parent / "blazeface-pytorch"
sys.path.insert(0, str(BLAZEFACE_DIR))
sys.path.insert(0, str(RETINA_DIR))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from blazeface import BlazeFace  # noqa: E402
import widerface_eval as we  # noqa: E402

DATASET_FOLDER = str(RETINA_DIR / "data/widerface/val/images/")
VAL_LIST = str(RETINA_DIR / "data/widerface/val/wider_val.txt")
GT_DIR = str(RETINA_DIR / "widerface_evaluation/ground_truth")


def generate_blazeface_predictions(model: BlazeFace, device: torch.device, save_folder: Path,
                                   min_score_thresh: float = 0.01, log_every: int = 200) -> None:
    """Runs BlazeFace over every WIDER FACE val image and writes predictions
    in the same per-image .txt format the rest of this repo's eval pipeline
    expects. min_score_thresh is set far below BlazeFace's own default
    (0.75, tuned for a live-camera single-best-face use case) so the
    resulting confidence distribution is wide enough for voc_ap's
    precision-recall sweep to be meaningful -- mirrors why
    generate_widerface_predictions uses conf_threshold=0.02 for RetinaFace
    instead of a high fixed cutoff."""
    import shutil

    if save_folder.exists():
        shutil.rmtree(save_folder)

    model.min_score_thresh = min_score_thresh

    with open(VAL_LIST, "r") as f:
        all_images = [name.lstrip("/") for name in f.read().split()]

    total = len(all_images)
    for done, img_name in enumerate(all_images, start=1):
        img_bgr = cv2.imread(str(Path(DATASET_FOLDER) / img_name), cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = img_rgb.shape[:2]
        resized = cv2.resize(img_rgb, (128, 128), interpolation=cv2.INTER_LINEAR)

        detections = model.predict_on_image(resized)  # (N, 17): ymin,xmin,ymax,xmax,...,score
        detections = detections.cpu().numpy() if isinstance(detections, torch.Tensor) else detections

        save_name = save_folder / (img_name[:-4] + ".txt")
        save_name.parent.mkdir(parents=True, exist_ok=True)
        with open(save_name, "w") as fd:
            fd.write(save_name.name[:-4] + "\n")
            fd.write(f"{len(detections)}\n")
            for det in detections:
                ymin, xmin, ymax, xmax = det[0], det[1], det[2], det[3]
                score = det[16]
                x = int(xmin * orig_w)
                y = int(ymin * orig_h)
                w = int(xmax * orig_w) - x
                h = int(ymax * orig_h) - y
                fd.write(f"{x} {y} {w} {h} {float(score)}\n")

        if done % log_every == 0 or done == total:
            print(f"[eval_blazeface] {done}/{total} images", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--min-score-thresh", type=float, default=0.01)
    p.add_argument("--pred-dir", default=str(RETINA_DIR / "results/blazeface/pred_full"))
    args = p.parse_args()

    device = torch.device(args.device)
    model = BlazeFace(back_model=False).to(device)
    model.load_weights(str(BLAZEFACE_DIR / "blazeface.pth"))
    model.load_anchors(str(BLAZEFACE_DIR / "anchors.npy"))
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    weights_mb = (BLAZEFACE_DIR / "blazeface.pth").stat().st_size / 1024 / 1024
    print(f"[eval_blazeface] params={n_params:,} weights_file={weights_mb:.3f}MB "
          f"device={args.device} min_score_thresh={args.min_score_thresh}")

    pred_dir = Path(args.pred_dir)
    pred_dir.mkdir(parents=True, exist_ok=True)

    generate_blazeface_predictions(model, device, pred_dir, min_score_thresh=args.min_score_thresh)
    aps = we.run_widerface_evaluation(str(pred_dir), GT_DIR)
    mean = we.mean_ap(aps)

    print("\n=== BlazeFace (front, 128x128) on WIDER FACE val ===")
    print(f"easy={aps['easy']:.4f}  medium={aps['medium']:.4f}  hard={aps['hard']:.4f}  mean={mean:.4f}")
    print(f"params={n_params:,}  weights_file={weights_mb:.3f}MB")


if __name__ == "__main__":
    main()
