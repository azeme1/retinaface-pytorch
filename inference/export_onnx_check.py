"""Checks an ONNX export from export_onnx.py against the PyTorch model it
came from -- both a fast numeric-parity spot check (raw box/score/landmark
tensors on a handful of images) and the real WIDER FACE AP (easy/medium/hard),
computed independently for the PyTorch wrapper and the ONNX Runtime session
over the SAME fixed-size preprocessing (this project's own resize/decode
machinery in widerface_eval.py assumes RetinaFace's own variable-size
aspect-preserving resize, which RetinaStaticExportWrapper deliberately does
NOT use -- see its docstring -- so AP here is only meaningful as an apples-
to-apples PyTorch-vs-ONNX comparison at THIS export's fixed image_size, not
as a number to compare against results/<network>/full_eval_parallel/*.json).

Unlike export_coreml_check.py's CoreML counterpart, onnxruntime has no
platform restriction -- this runs the ONNX side for real on any machine,
including this one.

Usage:
    python inference/export_onnx.py --network mobilenetv1 \\
        --checkpoint pytorch_export/mobilenetv1_c7.zip --image-size 640 --out-dir onnx_export
    python inference/export_onnx_check.py --network mobilenetv1 \\
        --checkpoint pytorch_export/mobilenetv1_c7.zip \\
        --onnx-zip onnx_export/mobilenetv1_c7.zip --image-size 640 --n-images 200

    # or pull both the pytorch source and the onnx artifact straight from a
    # published HF repo instead of local files (see export_pytorch_batch_hf.py):
    python inference/export_onnx_check.py --network mobilenetv1 \\
        --hf-repo azemel/retinaface-xs --hf-level c7 --image-size 640 --n-images 200
"""
import argparse
import os
import random
import shutil
import sys
import tempfile
import threading
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import tqdm

import cv2
import numpy as np
import onnxruntime as ort
import torch

_RETINA_DIR = Path(__file__).resolve().parents[1]
if str(_RETINA_DIR) not in sys.path:
    sys.path.append(str(_RETINA_DIR))

from config import get_config  # noqa: E402
import widerface_eval as we  # noqa: E402

sys.path.append(str(Path(__file__).resolve().parent))
from export_common import (  # noqa: E402
    RetinaStaticExportWrapper, load_plain_with_clusters, load_plain_with_clusters_from_hf,
    download_hf_artifact, select_device,
)

sys.path.append(str(_RETINA_DIR / "examples"))
from _postprocess import postprocess, rescale_to_original  # noqa: E402

DATASET_FOLDER = str(_RETINA_DIR / "data/widerface/val/images/")
VAL_LIST = str(_RETINA_DIR / "data/widerface/val/wider_val.txt")
GT_DIR = str(_RETINA_DIR / "widerface_evaluation/ground_truth")


def preprocess(img_bgr: np.ndarray, image_size: int) -> np.ndarray:
    """Fixed-shape stretch-resize matching RetinaStaticExportWrapper's own
    assumption (no aspect-preserving letterboxing -- see its docstring).
    Returns 1,3,H,W float32 BGR (this repo's own export/training order --
    export_onnx.py's graph has no input-color-order handling, unlike
    export_coreml.py's native-RGB-image path, since ONNX has no built-in
    image type: the raw tensor IS the contract, so it stays BGR here)."""
    resized = cv2.resize(img_bgr, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    return np.float32(resized).transpose(2, 0, 1)[None]




def print_ap_report(aps: dict) -> None:
    """Traditional WIDER FACE benchmark reporting: Easy/Medium/Hard/Average
    AP as percentages (the convention used across WIDER FACE leaderboards
    and papers), not raw 0-1 fractions."""
    print(f"  Easy:    {aps['easy'] * 100:6.2f}%")
    print(f"  Medium:  {aps['medium'] * 100:6.2f}%")
    print(f"  Hard:    {aps['hard'] * 100:6.2f}%")
    print(f"  Average: {we.mean_ap(aps) * 100:6.2f}%")


def print_comparison_table(network: str, format_name: str, pt_aps: dict, other_aps: dict) -> None:
    """Side-by-side PyTorch-vs-converted-format table -- the actual point
    of this whole script: not just two separate AP reports, but how much
    (if any) the conversion cost."""
    print(f"\n=== {network}: PyTorch vs {format_name} ===")
    print(f"{'':<10}{'PyTorch':>10}{format_name:>10}{'Diff':>10}")
    for label, key in [("Easy", "easy"), ("Medium", "medium"), ("Hard", "hard")]:
        pt_v, o_v = pt_aps[key] * 100, other_aps[key] * 100
        print(f"{label:<10}{pt_v:>9.2f}%{o_v:>9.2f}%{o_v - pt_v:>+9.2f}%")
    pt_avg, o_avg = we.mean_ap(pt_aps) * 100, we.mean_ap(other_aps) * 100
    print(f"{'Average':<10}{pt_avg:>9.2f}%{o_avg:>9.2f}%{o_avg - pt_avg:>+9.2f}%")


def write_prediction(save_folder: Path, img_name: str, boxes, scores):
    save_name = save_folder / (img_name[:-4] + ".txt")
    save_name.parent.mkdir(parents=True, exist_ok=True)
    with open(save_name, "w") as fd:
        fd.write(save_name.name[:-4] + "\n")
        fd.write(f"{len(boxes)}\n")
        for box, score in zip(boxes, scores.reshape(-1)):
            x, y = int(box[0]), int(box[1])
            w, h = int(box[2]) - x, int(box[3]) - y
            fd.write(f"{x} {y} {w} {h} {float(score)}\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--network", required=True)
    p.add_argument("--checkpoint", default=None, help="local export_pytorch.py output -- combined .zip or "
                                                        "bare .pth (see export_onnx.py's own --checkpoint) -- "
                                                        "mutually exclusive with --hf-repo")
    p.add_argument("--onnx-zip", default=None, help="local export_onnx.py output (.zip or raw .onnx) -- "
                                                      "mutually exclusive with --hf-repo")
    p.add_argument("--hf-repo", default=None, help="Hugging Face repo id (e.g. azemel/retinaface-xs) to "
                                                     "pull BOTH the pytorch source and the onnx artifact "
                                                     "from instead of local files -- requires --hf-level")
    p.add_argument("--hf-level", default=None, help="e.g. 'c7' or 'float32' (see "
                                                      "export_pytorch_batch_hf.py's tag convention) -- "
                                                      "selects results/<network>/{pytorch,onnx}/<network>_<level>.zip")
    p.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"),
                    help="bearer token for a private --hf-repo -- defaults to the HF_TOKEN system variable")
    p.add_argument("--image-size", type=int, default=640)
    p.add_argument("--n-images", type=int, default=None,
                    help="WIDER FACE val images to sample -- default is the FULL val set (3226 images), the "
                         "only way to get an AP comparable across runs; pass a smaller number for a quick, "
                         "noisier spot check instead")
    args = p.parse_args()

    cfg = dict(get_config(args.network))

    if args.hf_repo:
        assert args.hf_level, "--hf-repo needs --hf-level"
        assert not args.checkpoint and not args.onnx_zip, (
            "--hf-repo is mutually exclusive with --checkpoint/--onnx-zip"
        )
        model, _ = load_plain_with_clusters_from_hf(cfg, args.hf_repo, args.network, args.hf_level,
                                                     token=args.hf_token)
        onnx_zip = download_hf_artifact(args.hf_repo, args.network, "onnx", args.hf_level, token=args.hf_token)
    else:
        assert args.checkpoint and args.onnx_zip, (
            "need --checkpoint + --onnx-zip (or --hf-repo + --hf-level instead)"
        )
        model, _ = load_plain_with_clusters(cfg, args.checkpoint)
        onnx_zip = Path(args.onnx_zip)

    wrapper = RetinaStaticExportWrapper(model, cfg, image_size=(args.image_size, args.image_size)).eval()

    device = select_device()
    print(f"PyTorch reference running on: {device}")
    wrapper = wrapper.to(device)

    tmp_ctx = None
    if str(onnx_zip).endswith(".zip"):
        tmp_ctx = tempfile.TemporaryDirectory()
        with zipfile.ZipFile(onnx_zip) as zf:
            zf.extract("model.onnx", tmp_ctx.name)
        onnx_path = str(Path(tmp_ctx.name) / "model.onnx")
    else:
        onnx_path = str(onnx_zip)

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    with open(VAL_LIST) as f:
        all_images = [n.lstrip("/") for n in f.read().split()]
    sample = all_images if args.n_images is None else random.Random(0).sample(all_images, args.n_images)

    pt_pred_dir = _RETINA_DIR / "results" / args.network / "onnx_check_pred_pytorch"
    onnx_pred_dir = _RETINA_DIR / "results" / args.network / "onnx_check_pred_onnx"
    for d in (pt_pred_dir, onnx_pred_dir):
        if d.exists():
            shutil.rmtree(d)
        # see export_coreml_check.py's identical comment: every event in the
        # official GT list needs a prediction file, or run_widerface_evaluation
        # KeyErrors looking one up -- placeholder every val image first, only
        # the sampled subset gets overwritten with real predictions below.
        for img_name in all_images:
            write_prediction(d, img_name, np.zeros((0, 4)), np.zeros((0,)))

    diffs = {"boxes": [], "scores": [], "landmarks": []}
    session_lock_free = True  # onnxruntime CPUExecutionProvider sessions are thread-safe for .run()

    # MPS (Apple's own GPU backend) has a history of thread-safety issues
    # under concurrent inference from multiple Python threads -- confirmed
    # the analogous problem for CoreML's MLModel.predict() in
    # export_coreml_check.py (hangs at 0% under a ThreadPoolExecutor without
    # a lock), so guard MPS the same way defensively. CUDA's
    # concurrent-inference-from-multiple-threads path is well-established,
    # so it's left unlocked.
    pytorch_lock = threading.Lock() if device.type == "mps" else None

    def run_pytorch(pt_input: np.ndarray):
        x = torch.from_numpy(pt_input).to(device)
        with torch.no_grad():
            if pytorch_lock is not None:
                with pytorch_lock:
                    boxes, scores, landmarks = wrapper(x)
            else:
                boxes, scores, landmarks = wrapper(x)
        return boxes.cpu().numpy(), scores.cpu().numpy(), landmarks.cpu().numpy()

    def run_onnx(pt_input: np.ndarray):
        boxes, scores, landmarks = session.run(None, {input_name: pt_input})
        return boxes, scores, landmarks

    def process(name: str):
        img_bgr = cv2.imread(str(Path(DATASET_FOLDER) / name), cv2.IMREAD_COLOR)
        if img_bgr is None:
            return None
        native_h, native_w = img_bgr.shape[:2]
        pt_input = preprocess(img_bgr, args.image_size)

        pt_boxes, pt_scores, pt_landm = run_pytorch(pt_input)
        d_boxes, d_scores, d_landm = postprocess(pt_boxes, pt_scores, pt_landm)
        d_boxes, d_landm = rescale_to_original(d_boxes, d_landm, args.image_size, native_w, native_h)
        write_prediction(pt_pred_dir, name, d_boxes, d_scores)

        onnx_boxes, onnx_scores, onnx_landm = run_onnx(pt_input)
        o_boxes, o_scores, o_landm = postprocess(onnx_boxes, onnx_scores, onnx_landm)
        o_boxes, o_landm = rescale_to_original(o_boxes, o_landm, args.image_size, native_w, native_h)
        write_prediction(onnx_pred_dir, name, o_boxes, o_scores)

        return {
            "boxes": float(np.abs(pt_boxes - onnx_boxes).max()),
            "scores": float(np.abs(pt_scores - onnx_scores).max()),
            "landmarks": float(np.abs(pt_landm - onnx_landm).max()),
        }

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(process, name): name for name in sample}
        for fut in tqdm.tqdm(as_completed(futures), total=len(futures), desc="PyTorch vs ONNX"):
            parity = fut.result()
            if parity is None:
                continue
            for k in diffs:
                diffs[k].append(parity[k])

    if len(sample) < len(all_images):
        print(f"\nNote: only {len(sample)}/{len(all_images)} val images were actually run -- every other image "
              f"counts as 0 recall (same downward bias as this repo's own training-time quick-probe), so the "
              f"absolute AP below is NOT comparable to results/<network>/full_eval_parallel/*.json. The bias is "
              f"identical on both the PyTorch and ONNX sides, sampled from the same fixed seed, so the "
              f"PyTorch-vs-ONNX COMPARISON is still meaningful -- omit --n-images for the full val set instead.")

    print(f"\n=== {args.network}: PyTorch (fixed-size wrapper) real WIDER FACE AP, n={len(sample)} images ===")
    pt_aps = we.run_widerface_evaluation(str(pt_pred_dir), GT_DIR)
    print_ap_report(pt_aps)

    print(f"\n=== {args.network}: ONNX Runtime ({onnx_zip}) real WIDER FACE AP, n={len(sample)} images ===")
    onnx_aps = we.run_widerface_evaluation(str(onnx_pred_dir), GT_DIR)
    print_ap_report(onnx_aps)

    print_comparison_table(args.network, "ONNX", pt_aps, onnx_aps)

    print("\n=== numeric parity (raw tensors, before NMS/thresholding) ===")
    for k, vals in diffs.items():
        print(f"{k}: max={max(vals):.3e} mean={sum(vals)/len(vals):.3e}")

    if tmp_ctx is not None:
        tmp_ctx.cleanup()


if __name__ == "__main__":
    main()
