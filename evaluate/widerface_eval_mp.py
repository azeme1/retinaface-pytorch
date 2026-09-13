"""evaluate/'s own copy of the reusable pieces of ../widerface_eval.py --
generate_widerface_predictions, run_widerface_evaluation, evaluate_model,
mean_ap -- self-contained (drops load_retina_train_module, which exists
only for train_lut_quant.py and isn't used by worker.py/compare.py/
run_parallel.py anyway) and using multiprocessing for the CPU-bound work
instead of threading.

Why multiprocessing here specifically, when ../widerface_eval.py's own
docstring documents multiprocessing being tried and reverted for being
SLOWER: that measurement was for a design that forked a full process pool
AFTER the backbone's CUDA context was already initialized (unsafe -- fork-
after-CUDA-init hazard) and shipped each task a full decoded image array
(tens of MB) across a pipe. This module avoids both problems by splitting
the per-image work differently: the backbone forward pass stays serial, in
THIS process, on `device` (GPU/MPS/CPU) -- never forked after CUDA/MPS
init -- and only the decode/NMS/write step (already-CPU numpy arrays: one
image's raw loc/conf/landmarks plus a few scalars, not a full image) goes
to a ProcessPoolExecutor. That decode step is also always CPU-only
regardless of `device` (see box_utils.py's _MAX_EXP_INPUT comment and
inference/export_coreml.py's apply_palette_selective for the class of
accelerator-specific numerical bugs this sidesteps) -- pure-Python/numpy
work that's GIL-bound under threads, unlike the original file's on-device
decode, where CUDA/MPS calls released the GIL and gave ThreadPoolExecutor
real overlap. Real OS processes (separate GIL each) are what actually
parallelizes this now.
"""
import random
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

RETINA_DIR = Path(__file__).resolve().parent.parent
WIDERFACE_EVAL_DIR = RETINA_DIR / "widerface_evaluation"
INFER_DIR = RETINA_DIR / "inference"

for _p in (RETINA_DIR, WIDERFACE_EVAL_DIR, INFER_DIR):
    if str(_p) not in sys.path:
        sys.path.append(str(_p))

from layers.functions.prior_box import PriorBoxVectorized  # noqa: E402
from utils.box_utils import decode, decode_landmarks, nms  # noqa: E402
from evaluate_widerface import resize_image  # noqa: E402
from export_common import RetinaBackboneWrapper  # noqa: E402


def _memory_snapshot(device: torch.device) -> dict:
    """Cheap, dependency-free memory readout -- see ../widerface_eval.py's
    own version of this, unchanged here."""
    import resource
    snap = {"rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024}
    if device.type == "cuda":
        snap["gpu_allocated_mb"] = torch.cuda.memory_allocated(device) / 1e6
        snap["gpu_reserved_mb"] = torch.cuda.memory_reserved(device) / 1e6
    return snap


def _decode_and_write(img_name: str, loc: np.ndarray, conf: np.ndarray, landmarks: np.ndarray,
                       cfg: dict, img_height: int, img_width: int, resize_factor: float,
                       save_folder: str, conf_threshold: float, nms_threshold: float,
                       pre_nms_topk: int) -> str:
    """Pure CPU/numpy: PriorBox generation, decode, NMS, and writing this
    one image's prediction file -- everything needed here is already a
    plain numpy array or primitive (no torch model, no device), which is
    exactly what makes this cheap to pickle across the process boundary
    and safe to run in a worker process with no CUDA/MPS context of its
    own. A top-level (not nested/closure) function, required for
    ProcessPoolExecutor to pickle it under macOS/Windows' spawn start
    method."""
    priorbox = PriorBoxVectorized(cfg, image_size=(img_height, img_width))
    priors = priorbox.generate_anchors()

    boxes = decode(torch.from_numpy(loc), priors, cfg["variance"])
    landmarks_dec = decode_landmarks(torch.from_numpy(landmarks), priors, cfg["variance"])

    bbox_scale = torch.tensor([img_width, img_height] * 2)
    boxes = (boxes * bbox_scale / resize_factor).numpy()
    landmark_scale = torch.tensor([img_width, img_height] * 5)
    landmarks_out = (landmarks_dec * landmark_scale / resize_factor).numpy()

    scores = conf[:, 1]
    inds = scores > conf_threshold
    boxes, landmarks_out, scores = boxes[inds], landmarks_out[inds], scores[inds]

    order = scores.argsort()[::-1][:pre_nms_topk]
    boxes, landmarks_out, scores = boxes[order], landmarks_out[order], scores[order]

    detections = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = nms(detections, nms_threshold)
    detections = detections[keep]

    save_name = Path(save_folder) / (img_name[:-4] + ".txt")
    save_name.parent.mkdir(parents=True, exist_ok=True)
    with open(save_name, "w") as fd:
        fd.write(save_name.name[:-4] + "\n")
        fd.write(f"{len(detections)}\n")
        for box in detections:
            x, y = int(box[0]), int(box[1])
            w, h = int(box[2]) - x, int(box[3]) - y
            fd.write(f"{x} {y} {w} {h} {box[4]}\n")
    return img_name


@torch.no_grad()
def generate_widerface_predictions(model: nn.Module, cfg: dict, device: torch.device,
                                    dataset_folder: str, val_list_path: str, save_folder: str,
                                    conf_threshold: float = 0.02, nms_threshold: float = 0.4,
                                    origin_size: bool = False, sample_size: int | None = None,
                                    sample_seed: int = 0, pre_nms_topk: int = 5000,
                                    num_workers: int = 2, parallel_batch_size: int | None = None,
                                    log_progress: bool = True, log_every: int = 10) -> None:
    """Same contract and per-image pipeline as ../widerface_eval.py's own
    generate_widerface_predictions (see its docstring for sample_size/
    pre_nms_topk's rationale, unchanged here) -- only the parallelism
    mechanism differs, see this module's own docstring."""
    save_folder = Path(save_folder)
    if save_folder.exists():
        shutil.rmtree(save_folder)

    # Stage 1 (see inference/export_common.py's RetinaBackboneWrapper) --
    # normalization + backbone/heads only, no decode. Runs on `device`;
    # takes the raw BGR uint8 image straight from cv2.imread, no manual
    # mean-subtraction here anymore.
    backbone = RetinaBackboneWrapper(model, input_color_order="bgr").to(device).eval()

    with open(val_list_path) as f:
        all_images = [name.lstrip("/") for name in f.read().split()]

    if sample_size is not None and sample_size < len(all_images):
        for img_name in all_images:
            save_name = save_folder / (img_name[:-4] + ".txt")
            save_name.parent.mkdir(parents=True, exist_ok=True)
            with open(save_name, "w") as fd:
                fd.write(save_name.name[:-4] + "\n0\n")
        rng = random.Random(sample_seed)
        test_dataset = rng.sample(all_images, sample_size)
    else:
        test_dataset = all_images

    if parallel_batch_size is None:
        parallel_batch_size = int(cfg.get("batch_size", 1))

    total = len(test_dataset)
    done = 0

    def _log_progress():
        if not log_progress:
            return
        snap = _memory_snapshot(device)
        parts = [f"rss={snap['rss_mb']:.0f}MB"]
        if "gpu_allocated_mb" in snap:
            parts.append(f"gpu_alloc={snap['gpu_allocated_mb']:.0f}MB")
            parts.append(f"gpu_reserved={snap['gpu_reserved_mb']:.0f}MB")
        print(f"[widerface_eval_mp] {done}/{total} images -- " + " ".join(parts), flush=True)

    @torch.no_grad()
    def _run_backbone(img_name: str):
        """Main-process only: image decode/resize + the Stage 1 backbone
        forward pass on `device` -- returns everything _decode_and_write
        needs as plain numpy/primitives, small enough to pickle to a
        worker cheaply."""
        img_raw = cv2.imread(str(Path(dataset_folder) / img_name), cv2.IMREAD_COLOR)  # BGR, uint8

        resize_factor = 1 if origin_size else resize_image(img_raw)
        if resize_factor != 1:
            img_raw = cv2.resize(img_raw, None, None, fx=resize_factor, fy=resize_factor,
                                  interpolation=cv2.INTER_LINEAR)

        img_height, img_width, _ = img_raw.shape
        image = img_raw.astype(np.uint8).transpose(2, 0, 1)
        image_t = torch.from_numpy(image).unsqueeze(0).to(device)

        loc, conf, landmarks = backbone(image_t)
        loc = loc.cpu().numpy()
        conf = conf.cpu().numpy()
        landmarks = landmarks.cpu().numpy()
        return loc, conf, landmarks, img_height, img_width, resize_factor

    with ProcessPoolExecutor(max_workers=max(1, num_workers)) as pool:
        for i in range(0, len(test_dataset), parallel_batch_size):
            chunk = test_dataset[i:i + parallel_batch_size]
            futures = {}
            for img_name in chunk:
                loc, conf, landmarks, img_height, img_width, resize_factor = _run_backbone(img_name)
                fut = pool.submit(_decode_and_write, img_name, loc, conf, landmarks, cfg,
                                   img_height, img_width, resize_factor, str(save_folder),
                                   conf_threshold, nms_threshold, pre_nms_topk)
                futures[fut] = img_name
            for fut in as_completed(futures):
                fut.result()  # raises in this process if the worker raised
                done += 1
                if done % log_every == 0 or done == total:
                    _log_progress()


def _eval_event(setting_id: int, i: int, pred_dir: str, gt_dir: str):
    """Top-level (picklable) per-(setting, event) AP-curve contribution --
    see ../widerface_eval.py's run_widerface_evaluation for the original,
    thread-based version of this same computation. Re-imports the eval
    helpers and re-derives the gt/pred lists per call rather than closing
    over them, since ProcessPoolExecutor needs picklable args, not shared
    in-process state."""
    from evaluation import get_gt_boxes, get_preds, norm_score, image_eval, img_pr_info
    import numpy as np

    pred = get_preds(pred_dir)
    norm_score(pred)
    facebox_list, event_list, file_list, hard_gt_list, medium_gt_list, easy_gt_list = get_gt_boxes(gt_dir)
    setting_gts = [easy_gt_list, medium_gt_list, hard_gt_list]
    thresh_num = 1000

    gt_list = setting_gts[setting_id]
    event_name = str(event_list[i][0][0])
    img_list = file_list[i][0]
    pred_list = pred[event_name]
    sub_gt_list = gt_list[i][0]
    gt_bbx_list = facebox_list[i][0]

    local_curve = np.zeros((thresh_num, 2)).astype("float")
    local_count = 0
    for j in range(len(img_list)):
        pred_info = pred_list[str(img_list[j][0][0])]
        gt_boxes = gt_bbx_list[j][0].astype("float")
        keep_index = sub_gt_list[j][0]
        local_count += len(keep_index)

        if len(gt_boxes) == 0 or len(pred_info) == 0:
            continue
        ignore = np.zeros(gt_boxes.shape[0])
        if len(keep_index) != 0:
            ignore[keep_index - 1] = 1
        pred_recall, proposal_list = image_eval(pred_info, gt_boxes, ignore, 0.5)
        local_curve += img_pr_info(thresh_num, pred_info, proposal_list, pred_recall)
    return setting_id, local_curve, local_count


def run_widerface_evaluation(pred_dir: str, gt_dir: str, num_workers: int = 8) -> dict:
    """Same contract as ../widerface_eval.py's run_widerface_evaluation --
    ProcessPoolExecutor instead of ThreadPoolExecutor (that file's own
    docstring already notes a process pool would have been safe here too,
    since this touches no torch/CUDA state at all -- purely reading
    prediction .txt files off disk)."""
    from evaluation import get_gt_boxes, get_preds, norm_score
    import numpy as np

    pred = get_preds(pred_dir)
    norm_score(pred)
    _, event_list, *_ = get_gt_boxes(gt_dir)
    event_num = len(event_list)

    thresh_num = 1000
    pr_curves = [np.zeros((thresh_num, 2)).astype("float") for _ in range(3)]
    counts = [0, 0, 0]
    with ProcessPoolExecutor(max_workers=num_workers) as ex:
        futures = [ex.submit(_eval_event, setting_id, i, pred_dir, gt_dir)
                   for setting_id in range(3) for i in range(event_num)]
        for fut in as_completed(futures):
            setting_id, curve, count = fut.result()
            pr_curves[setting_id] += curve
            counts[setting_id] += count

    from evaluation import dataset_pr_info, voc_ap
    aps = []
    for setting_id in range(3):
        pr_curve = dataset_pr_info(thresh_num, pr_curves[setting_id], counts[setting_id])
        propose, recall = pr_curve[:, 0], pr_curve[:, 1]
        aps.append(voc_ap(recall, propose))

    return {"easy": aps[0], "medium": aps[1], "hard": aps[2]}


def evaluate_model(model: nn.Module, cfg: dict, device: torch.device, dataset_folder: str,
                    val_list_path: str, gt_dir: str, pred_dir: str, sample_size: int | None = None,
                    sample_seed: int = 0, num_workers: int = 2, parallel_batch_size: int | None = None,
                    log_progress: bool = True, log_every: int = 10, eval_num_workers: int = 8) -> dict:
    """generate_widerface_predictions + run_widerface_evaluation in one
    call -- same contract as ../widerface_eval.py's evaluate_model."""
    generate_widerface_predictions(model, cfg, device, dataset_folder, val_list_path, pred_dir,
                                    sample_size=sample_size, sample_seed=sample_seed,
                                    num_workers=num_workers, parallel_batch_size=parallel_batch_size,
                                    log_progress=log_progress, log_every=log_every)
    return run_widerface_evaluation(pred_dir, gt_dir, num_workers=eval_num_workers)


def mean_ap(aps: dict) -> float:
    return (aps["easy"] + aps["medium"] + aps["hard"]) / 3.0
