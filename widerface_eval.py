"""RetinaFace/WIDER-FACE-specific glue for train_lut_quant.py: loading this
repo's own train.py by file path (avoiding a name collision with the main
llwll repo's train/ package -- see load_retina_train_module's docstring),
and a programmatic (non-CLI) WIDER FACE evaluation harness used both for the
one-off float32 baseline measurement and for the per-epoch early-stopping
check during LUT-quantized fine-tuning. None of this is reusable outside
this repo -- the actually reusable quantization machinery
(fuse_bn_into_conv/apply_lut_quantization/estimate_compression_ratio) lives
in the main llwll repo's train/refine/quant_utils.py instead; this file only
ever imports FROM there, never contains a copy.

NOTE on a multiprocessing pool that USED to live here: per-image work
(cv2 decode/resize + PriorBox regeneration) was farmed out to a
ProcessPoolExecutor to use this box's many cores instead of one. Removed
after measuring it end-to-end: each worker had to be spawned fresh (fork
was unsafe -- the caller's CUDA context is already initialized by then) and
each task shipped a full decoded image array (tens of MB) plus its anchor
tensor back across a pipe. For this workload -- cheap per-image compute,
large per-item payloads, called repeatedly (once per QAT epoch) -- the
spawn + pickling overhead measured SLOWER in practice than the plain serial
loop (serial: ~90s for a 400-image probe, confirmed by direct timing;
pooled: 6+ minutes and still writing when killed). Kept as a cautionary
note rather than silently reintroducing the same regression later.

generate_widerface_predictions's num_workers/parallel_batch_size ARE a
second, later attempt at within-eval parallelism -- a ThreadPoolExecutor,
not a process pool, so it sidesteps both problems above (no fork-after-
CUDA-init, no cross-process pickling: everything stays in one process's
address space). Threads still contend on the GIL for the pure-Python NMS
loop, but cv2 decode/resize and the CUDA calls release it, so there's real
overlap available. Each chunk logs a memory snapshot (see _memory_snapshot)
so a leak across a long unsampled full-val pass -- or across several of
these processes running concurrently, as run_full_eval_parallel.py does --
shows up as a climbing curve instead of a silent OOM.
"""

import importlib.util
import sys
from pathlib import Path

import torch
import torch.nn as nn

RETINA_DIR = Path(__file__).resolve().parent
WIDERFACE_EVAL_DIR = RETINA_DIR / "widerface_evaluation"

for _p in (RETINA_DIR, WIDERFACE_EVAL_DIR):
    if str(_p) not in sys.path:
        sys.path.append(_p.as_posix())


def load_retina_train_module():
    """This repo's own train.py, loaded by explicit file path under its own
    sys.modules key ("retinaface_train_module") instead of the bare name
    "train" -- the main llwll repo (which train_lut_quant.py also imports
    from, for train/refine/quant_utils.py) has its OWN top-level train/
    PACKAGE, so a plain `import train` from this repo would be ambiguous
    with it (whichever sys.path entry Python resolves first for that bare
    name). train_one_epoch() inside the loaded module references `cfg` as a
    module global (only ever assigned in this file's own __main__ block), so
    callers must set `<returned module>.cfg = cfg_dict` before invoking it.
    """
    spec = importlib.util.spec_from_file_location("retinaface_train_module", RETINA_DIR / "train.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _memory_snapshot(device: torch.device) -> dict:
    """Cheap, dependency-free memory readout: process RSS via resource
    (ru_maxrss is Linux's PEAK RSS so far, in KB -- monotonically
    non-decreasing, which is actually the useful property here: a leak
    shows as a curve that keeps climbing chunk over chunk instead of
    plateauing) plus CUDA allocator stats when running on GPU. No psutil
    dependency -- resource is stdlib."""
    import resource
    snap = {"rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024}
    if device.type == "cuda":
        snap["gpu_allocated_mb"] = torch.cuda.memory_allocated(device) / 1e6
        snap["gpu_reserved_mb"] = torch.cuda.memory_reserved(device) / 1e6
    return snap


@torch.no_grad()
def generate_widerface_predictions(model: nn.Module, cfg: dict, device: torch.device,
                                   dataset_folder: str, val_list_path: str, save_folder: str,
                                   conf_threshold: float = 0.02, nms_threshold: float = 0.4,
                                   origin_size: bool = False, sample_size: int | None = None,
                                   sample_seed: int = 0, pre_nms_topk: int = 5000,
                                   num_workers: int = 2, parallel_batch_size: int | None = None,
                                   log_progress: bool = True, log_every: int = 10) -> None:
    """Same per-image pipeline as evaluate_widerface.py's main(), refactored
    to take an already-constructed, already-weight-loaded model instance
    instead of a (network name, weights path) pair -- evaluate_widerface.py
    always builds a plain RetinaFace(cfg=cfg) internally, which can't
    represent a fused+LUT-quantized model's actual module tree (Identity
    where BatchNorm2d used to be, QConv2d/QLinear instead of Conv2d/Linear),
    so reusing it as-is for anything past the float32 baseline isn't
    possible.

    sample_size caps how many val images actually get run through the model
    -- a full pass takes several minutes (dominated by per-image PriorBox
    regeneration at each image's own aspect-preserving resized resolution,
    not the forward pass itself), too slow to run after every QAT epoch for
    a same-epoch early-stopping check. widerface_evaluation/evaluation.py's
    own matching code indexes predictions by (event, image) pulled from the
    official .mat file's fixed event/image lists, so a naive "just run the
    first N images" subset would leave whole event folders (and images
    within a partially-covered folder) with no prediction file at all --
    evaluation() KeyErrors the moment it looks one up. Writing an empty
    (0-detection) placeholder for every val image first, then overwriting
    only the sampled subset with real inference, keeps every lookup valid
    while only paying real per-image cost for the sample -- the resulting
    AP is a downward-biased estimate of the true full-val AP (every
    unsampled image counts as 0 recall), but with sample_seed fixed across
    calls (the float32 baseline probe and every QAT epoch's probe use the
    same sample), the BIAS is identical each time, so the early-stopping
    comparison between them stays meaningful even though neither absolute
    number matches the real full-val AP -- only run with sample_size=None
    (a real full pass) for the numbers actually reported at the end.

    pre_nms_topk caps how many confidence-threshold-surviving candidates
    reach nms() per image (same convention as this repo's own
    onnx_inference.py, just missing from evaluate_widerface.py's own
    pipeline, which this function otherwise mirrors). Without it: nms()'s
    is an O(kept^2)-ish Python while loop, fine for the few hundred
    candidates a well-calibrated model produces, but a QAT-quantized model
    early in fine-tuning (confirmed: one epoch into 2-cluster QAT) can have
    badly miscalibrated confidence outputs -- observed 30,000+ candidates
    surviving conf_threshold on a single image, most not overlapping enough
    for NMS to suppress each other, turning a ~0.2s/image pass into
    several SECONDS per image (a full quick-probe going from ~90s to 10+
    minutes, confirmed by direct measurement -- this, not multiprocessing
    or CUDA allocator state, both tried and reverted first, was the actual
    cause of what looked like a hang during early QAT epochs).
    """
    import cv2
    import numpy as np
    import random
    import shutil

    from layers.functions.prior_box import PriorBoxVectorized
    from utils.box_utils import decode, decode_landmarks, nms
    from evaluate_widerface import resize_image

    save_folder = Path(save_folder)
    if save_folder.exists():
        shutil.rmtree(save_folder)

    rgb_mean = (104, 117, 123)
    model.eval()

    with open(val_list_path, "r") as f:
        all_images = f.read().split()

    # wider_val.txt entries start with "/" (e.g. "/24--Soldier_Firing/xxx.jpg") --
    # original evaluate_widerface.py string-concatenates them onto
    # dataset_folder; Path(dataset_folder) / img_name would instead treat
    # the leading "/" as an absolute path and silently discard
    # dataset_folder entirely (confirmed: raises reading from a bogus
    # root-relative path). Strip it and always join explicitly instead.
    all_images = [name.lstrip("/") for name in all_images]

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
    # Decide chunk size for parallel submissions; default to training batch size
    if parallel_batch_size is None:
        parallel_batch_size = int(cfg.get("batch_size", 1))

    from concurrent.futures import ThreadPoolExecutor, as_completed

    @torch.no_grad()
    def _process_image(img_name: str):
        # torch's grad-enabled flag is thread-local, so the outer function's
        # @torch.no_grad() (set in the calling thread) does NOT propagate into
        # ThreadPoolExecutor worker threads -- each spawns with grad enabled
        # by default, so model(image) output tensors come back requiring
        # grad and .cpu().numpy() below raises. Re-applied here so it's in
        # effect regardless of which thread actually runs this.
        try:
            img_raw = cv2.imread(str(Path(dataset_folder) / img_name), cv2.IMREAD_COLOR)
            image = np.float32(img_raw)

            resize_factor = 1 if origin_size else resize_image(image)
            if resize_factor != 1:
                image = cv2.resize(image, None, None, fx=resize_factor, fy=resize_factor, interpolation=cv2.INTER_LINEAR)

            img_height, img_width, _ = image.shape
            image -= rgb_mean
            image = image.transpose(2, 0, 1)
            image = torch.from_numpy(image).unsqueeze(0).to(device)

            loc, conf, landmarks = model(image)
            loc, conf, landmarks = loc.squeeze(0), conf.squeeze(0), landmarks.squeeze(0)

            priorbox = PriorBoxVectorized(cfg, image_size=(img_height, img_width))
            priors = priorbox.generate_anchors().to(device)

            boxes = decode(loc, priors, cfg["variance"])
            landmarks = decode_landmarks(landmarks, priors, cfg["variance"])

            bbox_scale = torch.tensor([img_width, img_height] * 2, device=device)
            boxes = (boxes * bbox_scale / resize_factor).cpu().numpy()
            landmark_scale = torch.tensor([img_width, img_height] * 5, device=device)
            landmarks = (landmarks * landmark_scale / resize_factor).cpu().numpy()

            scores = conf.cpu().numpy()[:, 1]
            inds = scores > conf_threshold
            boxes, landmarks, scores = boxes[inds], landmarks[inds], scores[inds]

            order = scores.argsort()[::-1][:pre_nms_topk]
            boxes, landmarks, scores = boxes[order], landmarks[order], scores[order]

            detections = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
            keep = nms(detections, nms_threshold)
            detections = detections[keep]

            save_name = save_folder / (img_name[:-4] + ".txt")
            save_name.parent.mkdir(parents=True, exist_ok=True)
            with open(save_name, "w") as fd:
                fd.write(Path(save_name).name[:-4] + "\n")
                fd.write(f"{len(detections)}\n")
                for box in detections:
                    x, y = int(box[0]), int(box[1])
                    w, h = int(box[2]) - x, int(box[3]) - y
                    fd.write(f"{x} {y} {w} {h} {box[4]}\n")
            return None
        except Exception:
            # propagate exception details to main thread
            import traceback
            return traceback.format_exc()

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
        print(f"[widerface_eval] {done}/{total} images -- " + " ".join(parts), flush=True)

    if num_workers is None or num_workers <= 1:
        # fall back to serial processing -- still fills the output (one
        # prediction .txt per image, written as each finishes) incrementally,
        # and logs a memory snapshot every log_every images (independent of
        # parallel_batch_size, which only controls thread-submission chunk
        # size below) so a leak shows up as a climbing rss_mb/gpu_reserved_mb
        # curve well before it OOMs, not just as a crash at the very end.
        for img_name in test_dataset:
            err = _process_image(img_name)
            if err:
                raise RuntimeError(err)
            done += 1
            if done % log_every == 0 or done == total:
                _log_progress()
    else:
        # Process in chunks of size `parallel_batch_size`, submitting each chunk
        # to a ThreadPoolExecutor with `num_workers` concurrent threads. Memory
        # is logged every log_every completions, independent of chunk size, so
        # a small log_every still gives fine-grained monitoring even when
        # parallel_batch_size (e.g. the training batch size) is large.
        with ThreadPoolExecutor(max_workers=num_workers) as ex:
            for i in range(0, len(test_dataset), parallel_batch_size):
                chunk = test_dataset[i:i + parallel_batch_size]
                futures = {ex.submit(_process_image, name): name for name in chunk}
                for fut in as_completed(futures):
                    res = fut.result()
                    if res:
                        # res is traceback string
                        raise RuntimeError(f"Error processing {futures[fut]}:\n" + res)
                    done += 1
                    if done % log_every == 0 or done == total:
                        _log_progress()


def run_widerface_evaluation(pred_dir: str, gt_dir: str, num_workers: int = 8) -> dict:
    """Official Easy/Medium/Hard AP via widerface_evaluation/evaluation.py's
    own matching code (image_eval/img_pr_info/dataset_pr_info/voc_ap) --
    duplicated here only to add a return value; that script's own
    evaluation() prints the three AP numbers but returns None.

    Parallelized across (setting, event) pairs with a ThreadPoolExecutor --
    same choice as generate_widerface_predictions's own num_workers, and for
    the same reason: no fork-after-CUDA-init hazard, no cross-process
    pickling (this function doesn't touch torch/CUDA at all -- it only reads
    the .txt prediction files generate_widerface_predictions already wrote
    to disk -- so a ProcessPoolExecutor would actually have been safe here
    too, but staying consistent with the established in-process pattern is
    simpler). Each (setting, event) task is independent -- it only reads
    shared read-only data (pred/gt lists) and returns its own partial
    pr_curve/count_face contribution -- summed in the main thread after all
    complete, so no lock is needed anywhere in the accumulation.
    """
    from evaluation import get_gt_boxes, get_preds, norm_score, image_eval, img_pr_info, dataset_pr_info, voc_ap
    import numpy as np
    from concurrent.futures import ThreadPoolExecutor, as_completed

    pred = get_preds(pred_dir)
    norm_score(pred)
    facebox_list, event_list, file_list, hard_gt_list, medium_gt_list, easy_gt_list = get_gt_boxes(gt_dir)
    event_num = len(event_list)
    thresh_num = 1000
    setting_gts = [easy_gt_list, medium_gt_list, hard_gt_list]

    def _eval_event(setting_id: int, i: int):
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

    pr_curves = [np.zeros((thresh_num, 2)).astype("float") for _ in range(3)]
    counts = [0, 0, 0]
    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        futures = [ex.submit(_eval_event, setting_id, i)
                   for setting_id in range(3) for i in range(event_num)]
        for fut in as_completed(futures):
            setting_id, curve, count = fut.result()
            pr_curves[setting_id] += curve
            counts[setting_id] += count

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
    """generate_widerface_predictions + run_widerface_evaluation in one call.
    num_workers sizes the per-image (GPU-bound) prediction phase;
    eval_num_workers separately sizes the (setting, event) AP-computation
    phase -- the two have different ideal widths (the former is limited by
    GPU-context contention, the latter is pure CPU/numpy work), so they're
    not tied together."""
    generate_widerface_predictions(model, cfg, device, dataset_folder, val_list_path, pred_dir,
                                   sample_size=sample_size, sample_seed=sample_seed,
                                   num_workers=num_workers, parallel_batch_size=parallel_batch_size,
                                   log_progress=log_progress, log_every=log_every)
    return run_widerface_evaluation(pred_dir, gt_dir, num_workers=eval_num_workers)


def mean_ap(aps: dict) -> float:
    return (aps["easy"] + aps["medium"] + aps["hard"]) / 3.0
