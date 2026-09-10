"""Checks a CoreML export from export_coreml.py against the PyTorch model it
came from -- both a fast numeric-parity spot check (raw box/score/landmark
tensors on a handful of images) and the real WIDER FACE AP (easy/medium/hard),
computed independently for the PyTorch wrapper and the CoreML model over the
SAME fixed-size preprocessing (this project's own resize/decode machinery in
widerface_eval.py assumes RetinaFace's own variable-size aspect-preserving
resize, which RetinaStaticExportWrapper deliberately does NOT use -- see its
docstring -- so AP here is only meaningful as an apples-to-apples PyTorch-vs-
CoreML comparison at THIS export's fixed image_size, not as a number to
compare against results/<network>/full_eval_parallel/*.json).

Running the CoreML model needs coremltools' native runtime
(libcoremlpython), which only exists on macOS -- on any other platform this
script still builds/exports the PyTorch side and reports that clearly rather
than crashing on the missing runtime.

Usage:
    python inference/export_coreml.py --network resnet18 \\
        --checkpoint pytorch_export/resnet18_c2.zip --image-size 640 --out-dir coreml_export
    python inference/export_coreml_check.py --network resnet18 \\
        --checkpoint pytorch_export/resnet18_c2.zip \\
        --mlpackage coreml_export/resnet18_c2.zip \\
        --image-size 640 --n-images 60

    # or pull both the pytorch source and the coreml artifact straight from a
    # published HF repo instead of local files (see export_pytorch_batch_hf.py) --
    # this is the form to run on a Mac, where .predict() actually works:
    python inference/export_coreml_check.py --network resnet18 \\
        --hf-repo azemel/retinaface-xs --hf-level c2 --image-size 640 --n-images 60

    # or download either artifact from an arbitrary URL directly (e.g. a
    # private repo's own resolve/main/... link, needs --hf-token or $HF_TOKEN):
    python inference/export_coreml_check.py --network mobilenetv1 --image-size 640 \\
        --checkpoint-url https://huggingface.co/azemel/retinaface-xs/resolve/main/results/mobilenetv1/pytorch/mobilenetv1_c12.zip \\
        --mlpackage-url https://huggingface.co/azemel/retinaface-xs/resolve/main/results/mobilenetv1/coreml/mobilenetv1_c12.zip
"""
from __future__ import annotations

import argparse
import os
import random
import shutil
import sys
import tempfile
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import coremltools as ct
import cv2
import numpy as np
import PIL.Image
import torch

_RETINA_DIR = Path(__file__).resolve().parents[1]
if str(_RETINA_DIR) not in sys.path:
    sys.path.append(str(_RETINA_DIR))

from config import get_config  # noqa: E402
from utils.box_utils import nms  # noqa: E402
import widerface_eval as we  # noqa: E402

sys.path.append(str(Path(__file__).resolve().parent))
from export_common import (  # noqa: E402
    RetinaStaticExportWrapper, load_plain_with_clusters, load_plain_with_clusters_from_hf,
    download_hf_artifact, replace_leaky_relu, download_from_url,
)

DATASET_FOLDER = str(_RETINA_DIR / "data/widerface/val/images/")
VAL_LIST = str(_RETINA_DIR / "data/widerface/val/wider_val.txt")
GT_DIR = str(_RETINA_DIR / "widerface_evaluation/ground_truth")


def preprocess(img_bgr: np.ndarray, image_size: int, input_color_order: str) -> tuple[np.ndarray, np.ndarray]:
    """Fixed-shape stretch-resize matching RetinaStaticExportWrapper's own
    assumption (no aspect-preserving letterboxing -- see its docstring).
    Returns (pytorch_input: 1,3,H,W float32 in input_color_order, coreml_input:
    H,W,3 uint8 RGB -- CoreML's native image input is always declared RGB
    regardless of input_color_order, see export_coreml.py)."""
    resized_bgr = cv2.resize(img_bgr, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    resized_rgb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)

    ordered = resized_rgb if input_color_order == "rgb" else resized_bgr
    pt_input = np.float32(ordered).transpose(2, 0, 1)[None]  # 1,3,H,W
    cml_input = resized_rgb.astype(np.uint8)  # H,W,3 -- CoreML's own ImageType input
    return pt_input, cml_input


def top_detections(boxes, scores, landmarks, conf_threshold=0.5, nms_threshold=0.4, topk=50):
    inds = scores.reshape(-1) > conf_threshold
    boxes, scores, landmarks = boxes[inds], scores[inds], landmarks[inds]
    order = scores.reshape(-1).argsort()[::-1][:topk]
    boxes, scores, landmarks = boxes[order], scores[order], landmarks[order]
    dets = np.hstack((boxes, scores.reshape(-1, 1))).astype(np.float32)
    keep = nms(dets, nms_threshold)
    return boxes[keep], scores[keep], landmarks[keep]


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
                                                        "bare .pth (see export_coreml.py's own --checkpoint) -- "
                                                        "mutually exclusive with --checkpoint-url/--hf-repo")
    p.add_argument("--checkpoint-url", default=None, help="download --checkpoint from this URL instead of a "
                                                            "local file, e.g. a direct link into a Hugging Face "
                                                            "repo's resolve/main/... path")
    p.add_argument("--mlpackage", default=None, help="local export_coreml.py output (.zip or raw "
                                                       ".mlpackage) -- mutually exclusive with --mlpackage-url/--hf-repo")
    p.add_argument("--mlpackage-url", default=None, help="download --mlpackage from this URL instead of a "
                                                           "local file")
    p.add_argument("--hf-repo", default=None, help="Hugging Face repo id (e.g. azemel/retinaface-xs) to "
                                                     "pull BOTH the pytorch source and the coreml artifact "
                                                     "from instead of local files -- requires --hf-level")
    p.add_argument("--hf-level", default=None, help="e.g. 'c7' or 'float32' (see "
                                                      "export_pytorch_batch_hf.py's tag convention) -- "
                                                      "selects results/<network>/{pytorch,coreml}/<network>_<level>.zip")
    p.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"),
                    help="bearer token for a private repo -- defaults to the HF_TOKEN system variable, used for "
                         "--hf-repo as well as --checkpoint-url/--mlpackage-url")
    p.add_argument("--input-color-order", default="bgr", choices=["bgr", "rgb"],
                    help="must match what export_coreml.py used to build this .mlpackage")
    p.add_argument("--image-size", type=int, default=640)
    p.add_argument("--n-images", type=int, default=60, help="WIDER FACE val images to sample")
    args = p.parse_args()

    cfg = dict(get_config(args.network))

    def _resolve(local: str | None, url: str | None, label: str) -> str | None:
        assert not (local and url), f"--{label} and --{label}-url are mutually exclusive"
        return str(download_from_url(url, token=args.hf_token)) if url else local

    if args.hf_repo:
        assert args.hf_level, "--hf-repo needs --hf-level"
        assert not args.checkpoint and not args.checkpoint_url and not args.mlpackage and not args.mlpackage_url, (
            "--hf-repo is mutually exclusive with --checkpoint/--checkpoint-url/--mlpackage/--mlpackage-url"
        )
        model, _ = load_plain_with_clusters_from_hf(cfg, args.hf_repo, args.network, args.hf_level,
                                                     token=args.hf_token)
        mlpackage_arg = str(download_hf_artifact(args.hf_repo, args.network, "coreml", args.hf_level,
                                                  token=args.hf_token))
    else:
        checkpoint_arg = _resolve(args.checkpoint, args.checkpoint_url, "checkpoint")
        mlpackage_arg = _resolve(args.mlpackage, args.mlpackage_url, "mlpackage")
        assert checkpoint_arg and mlpackage_arg, (
            "need --checkpoint/--checkpoint-url + --mlpackage/--mlpackage-url (or --hf-repo + --hf-level instead)"
        )
        model, _ = load_plain_with_clusters(cfg, checkpoint_arg)

    wrapper = RetinaStaticExportWrapper(model, cfg, image_size=(args.image_size, args.image_size),
                                        priors_dtype=torch.float16,
                                        input_color_order=args.input_color_order).eval()
    replace_leaky_relu(wrapper)

    mlmodel = None
    can_run_coreml = False
    tmp_ctx = None
    mlpackage_source_label = mlpackage_arg  # kept for the printed report below -- mlpackage_arg
    # itself gets reassigned to a temp-extracted local path if this is a zip
    if mlpackage_arg.endswith(".zip"):
        # export_coreml.py's own zip packaging (see inference/export_pytorch_batch_hf.py's
        # convention): one generically-named "model.mlpackage" dir inside --
        # extract it once here so callers can pass either form.
        tmp_ctx = tempfile.TemporaryDirectory()
        with zipfile.ZipFile(mlpackage_arg) as zf:
            zf.extractall(tmp_ctx.name)
        mlpackage_arg = str(Path(tmp_ctx.name) / "model.mlpackage")
    try:
        mlmodel = ct.models.MLModel(mlpackage_arg)
        # Loading an .mlpackage (parsing its protobuf) succeeds on any
        # platform -- it's .predict() specifically that needs the native
        # macOS CoreML runtime (libcoremlpython). Probe that here, once,
        # rather than discovering it independently inside every image in
        # the thread pool below.
        dummy = np.zeros((args.image_size, args.image_size, 3), dtype=np.uint8)
        mlmodel.predict({"input": PIL.Image.fromarray(dummy, mode="RGB")})
        can_run_coreml = True
    except Exception as e:
        print(f"[warning] can't run this .mlpackage on this platform ({type(e).__name__}: {e}) -- "
              f"coremltools' .predict() needs the native macOS CoreML runtime (libcoremlpython), "
              f"not available here. Reporting PyTorch-only.")

    with open(VAL_LIST) as f:
        all_images = [n.lstrip("/") for n in f.read().split()]
    sample = random.Random(0).sample(all_images, args.n_images)

    pt_pred_dir = _RETINA_DIR / "results" / args.network / "coreml_check_pred_pytorch"
    cml_pred_dir = _RETINA_DIR / "results" / args.network / "coreml_check_pred_coreml"
    for d in (pt_pred_dir, cml_pred_dir):
        if d.exists():
            shutil.rmtree(d)
        # run_widerface_evaluation looks up every image in every WIDER FACE
        # event from the official fixed GT list -- an event with none of the
        # sampled images written yet would KeyError. Empty (0-detection)
        # placeholders for every val image first, exactly like
        # widerface_eval.generate_widerface_predictions's own sample_size
        # path, keep every lookup valid; only the sampled subset gets
        # overwritten with real predictions below.
        for img_name in all_images:
            write_prediction(d, img_name, np.zeros((0, 4)), np.zeros((0,)))

    def run_pytorch(pt_input: np.ndarray):
        with torch.no_grad():
            boxes, scores, landmarks = wrapper(torch.from_numpy(pt_input))
        return boxes.numpy(), scores.numpy(), landmarks.numpy()

    def run_coreml(cml_input: np.ndarray):
        img = PIL.Image.fromarray(cml_input, mode="RGB")
        out = mlmodel.predict({"input": img})
        return out["boxes"], out["scores"], out["landmarks"]

    diffs = {"boxes": [], "scores": [], "landmarks": []}

    def process(name: str):
        img_bgr = cv2.imread(str(Path(DATASET_FOLDER) / name), cv2.IMREAD_COLOR)
        if img_bgr is None:
            return None
        pt_input, cml_input = preprocess(img_bgr, args.image_size, args.input_color_order)

        pt_boxes, pt_scores, pt_landm = run_pytorch(pt_input)
        d_boxes, d_scores, d_landm = top_detections(pt_boxes, pt_scores, pt_landm)
        write_prediction(pt_pred_dir, name, d_boxes, d_scores)

        parity = None
        if can_run_coreml:
            cml_boxes, cml_scores, cml_landm = run_coreml(cml_input)
            c_boxes, c_scores, c_landm = top_detections(cml_boxes, cml_scores, cml_landm)
            write_prediction(cml_pred_dir, name, c_boxes, c_scores)
            parity = {
                "boxes": float(np.abs(pt_boxes - cml_boxes).max()),
                "scores": float(np.abs(pt_scores - cml_scores).max()),
                "landmarks": float(np.abs(pt_landm - cml_landm).max()),
            }
        return parity

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(process, name): name for name in sample}
        for fut in as_completed(futures):
            parity = fut.result()
            if parity is None:
                continue
            for k in diffs:
                diffs[k].append(parity[k])

    print(f"\nNote: only {len(sample)}/{len(all_images)} val images were actually run -- every other image "
          f"counts as 0 recall (same downward bias as this repo's own training-time quick-probe), so the "
          f"absolute AP below is NOT comparable to results/<network>/full_eval_parallel/*.json. The bias is "
          f"identical on both the PyTorch and CoreML sides, sampled from the same fixed seed, so the "
          f"PyTorch-vs-CoreML COMPARISON is still meaningful -- raise --n-images for a tighter one.")

    print(f"\n=== {args.network}: PyTorch (fixed-size wrapper) real WIDER FACE AP, n={len(sample)} images ===")
    pt_aps = we.run_widerface_evaluation(str(pt_pred_dir), GT_DIR)
    print(f"easy={pt_aps['easy']:.4f} medium={pt_aps['medium']:.4f} hard={pt_aps['hard']:.4f} "
          f"mean={we.mean_ap(pt_aps):.4f}")

    if can_run_coreml:
        print(f"\n=== {args.network}: CoreML ({mlpackage_source_label}) real WIDER FACE AP, n={len(sample)} images ===")
        cml_aps = we.run_widerface_evaluation(str(cml_pred_dir), GT_DIR)
        print(f"easy={cml_aps['easy']:.4f} medium={cml_aps['medium']:.4f} hard={cml_aps['hard']:.4f} "
              f"mean={we.mean_ap(cml_aps):.4f}")

        print(f"\n=== numeric parity (raw tensors, before NMS/thresholding) ===")
        for k, vals in diffs.items():
            print(f"{k}: max={max(vals):.3e} mean={sum(vals)/len(vals):.3e}")
    else:
        print("\n(CoreML-side AP and parity skipped -- see warning above)")

    if tmp_ctx is not None:
        tmp_ctx.cleanup()


if __name__ == "__main__":
    main()
