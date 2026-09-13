"""QAT fine-tune a single RetinaFace backbone with per-output-channel LUT
quantization, gradually annealed from a large cluster count down to a
small target (default 2), starting from this repo's own released
pretrained float32 checkpoint where one exists (mobilenetv1_0.25/0.50,
mobilenetv1, mobilenetv2, resnet18, resnet34). resnet50 has no released
checkpoint (README: "not available"), so it first gets a reduced-epoch
float32 baseline trained from scratch (ImageNet-backbone init, standard
train.py pipeline) before the same quantize+QAT-fine-tune procedure runs
on top of it. BatchNorm IS fused into the preceding conv, correctly, via
quantize_graph (see build_quantized_model's docstring) -- an earlier
version of this pipeline used a naive fuse-then-quantize that measurably
hurt quality; quantize_graph/quantize_fused_batch_norm is quant_utils.py's
rebuilt fix for that failure mode, so BatchNorm2d no longer exists as a
separate module after quantization -- its effect is folded into each
fused layer's own quantized weight/bias.

This file is deliberately thin: the actual quantization machinery
(quantize_graph/anneal_lut_clusters/estimate_compression_ratio) is
imported from the main llwll repo's
train/refine/quant_utils.py, not reimplemented here -- this repo only
supplies the RetinaFace-specific glue (model construction, checkpoint
loading, WIDER FACE evaluation, in widerface_eval.py) that a generic
model-agnostic helper can't know about. Same split is meant to be reused
for other vendored repos later: the reusable "ideas" live in llwll, each
vendored repo gets its own small driver like this one.

Pipeline, reusing this repo's own train.py's dataset/loss/optimizer at
every stage (train_one_epoch itself only for the resnet50 float32
baseline -- see train_one_epoch_clipped's docstring for why cluster
annealing needs its own, gradient-clipped variant instead):
    1. (resnet50 only) float32 baseline training, reduced epoch budget.
    2. Evaluate the float32 checkpoint on the REAL WIDER FACE val set
       (Easy/Medium/Hard AP) -- this run's quality reference.
    3. Evaluate the SAME checkpoint on a fixed-sample quick probe (same
       sample every epoch below) -- a fast, consistently-biased proxy AP
       used only for the early-stop comparison, not reported as a real
       number (see widerface_eval.generate_widerface_predictions's
       docstring).
    4. Walk DOWN --anneal-schedule (default 256,128,64,32,16,15,...,3) to
       --num-clusters (default 2), one level at a time. EVERY level -- not
       just the final target -- gets: quantize/
       re-estimate the LUT from the model's current weights (see
       anneal_lut_clusters), QAT fine-tune with the same early-stop-vs-
       float32-baseline check, and its own saved checkpoint
       (results_dir/{network}_lut{k}_checkpoint.pth). COARSE levels
       (num_clusters >= COARSE_FINE_THRESHOLD) get --anneal-epochs at
       constant LR; FINE levels (below it, including the final target) get
       the bigger --qat-epochs budget with real (scaled) milestone LR
       decay -- per-level early stopping still exits the moment a level
       matches baseline, so most levels use far less than their budget in
       practice. A one-shot jump straight to a small codebook
       (num_clusters=2 in particular) was confirmed to collapse training
       outright -- matches a failure mode this repo's own qwen/
       soft_quant.py documents fighting at the same cluster count, via a
       different mechanism (see anneal_lut_clusters's docstring).
    5. Pick the level to actually report (pick_best_level): the level with
       the HIGHEST measured compression_ratio among those whose quick-probe
       quality matched/beat the float32 baseline -- not the smallest
       num_clusters, since a high cluster count's own per-channel codebook
       can make the "compressed" model LARGER than float32 (confirmed:
       num_clusters=256 measured 0.43x on mobilenetv1_0.25). Falls back to
       the best-quality level among those that still compress at all if
       none met the baseline, and to the single best-quality level seen if
       nothing compresses either (never silently ship the most-collapsed
       level just because it was last). Rebuilds a model quantized at
       exactly that level's cluster count and loads its saved checkpoint.
    6. Evaluate the SELECTED level on the real val set the same way as
       step 2, and estimate its compression ratio.
    7. Write results_dir/report.json with every level's numbers, not just
       the selected one.

Usage:
    python train_lut_quant.py --network resnet34 \\
        --pretrained weights/retinaface_r34.pth --results-dir .../resnet34
    python train_lut_quant.py --network resnet50 --results-dir .../resnet50
"""

import sys
from pathlib import Path

RETINA_DIR = Path(__file__).resolve().parent  # already sys.path[0] (the interpreter adds a script's own dir)
REPO_ROOT = Path("/workspace/home_0/work/llwll")

# Must outrank RETINA_DIR for the bare name "train": this repo has its own
# top-level train.py, and the main repo has a train/ PACKAGE -- whichever is
# found first on sys.path wins that name. Inserting at 0 here still wins
# even though RETINA_DIR was auto-added at 0 before this line ran (the
# interpreter does that at startup, ahead of any of this file's own code).
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from train.refine.quant_utils import (  # noqa: E402
    quantize_graph, apply_lut_quantization, anneal_lut_clusters, estimate_compression_ratio,
)

import widerface_eval as we  # noqa: E402  -- this repo's own file; RETINA_DIR is already on sys.path

import argparse
import copy
import gc
import json
import os
import time

import torch
from torch.utils.data import DataLoader

from config import get_config
from models import RetinaFace
from layers import PriorBox, MultiBoxLoss
from utils.dataset import WiderFaceDetection
from utils.transform import Augmentation

retina_train = we.load_retina_train_module()  # this repo's own train.py, loaded by file path (see widerface_eval.py)

RGB_MEAN = (104, 117, 123)  # bgr order, same as train.py

DATASET_FOLDER = str(RETINA_DIR / "data/widerface/val/images/")
VAL_LIST = str(RETINA_DIR / "data/widerface/val/wider_val.txt")
TRAIN_DATA = str(RETINA_DIR / "data/widerface/train")
GT_DIR = str(RETINA_DIR / "widerface_evaluation/ground_truth")

QUICK_PROBE_SIZE = 400
QUICK_PROBE_SEED = 0

# widerface_eval.py's evaluate_model has TWO differently-scaling knobs, not
# one -- see evaluate_model's docstring: num_workers sizes a ThreadPoolExecutor
# doing the per-image GPU-bound forward pass (each thread holds its own
# activations on the SAME GPU concurrently), while eval_num_workers sizes a
# separate, purely CPU-bound (setting, event) AP-computation phase (NMS/decode,
# no GPU involved). Giving both the full core count (previously EVAL_NUM_WORKERS
# = os.cpu_count(), applied to both) was fine for narrow backbones but caused
# an immediate CUDA OOM on the full-width mobilenetv1/mobilenetv2 backbones --
# confirmed via traceback (OOM inside BatchNorm2d.forward, before any training
# epoch even started, i.e. during the initial float32 baseline eval) -- ~128
# concurrent forward passes' activations for a wider backbone don't fit in
# 16GB, regardless of --batch-size (which only sizes the training DataLoader,
# entirely separate from this eval path). EVAL_GPU_WORKERS stays small and
# backbone-size-independent (matches evaluate/worker.py's own conservative
# num_workers=2 default for a single eval, given headroom since this script
# only ever runs ONE eval at a time, not many concurrent ones); EVAL_CPU_WORKERS
# keeps the full core count since that phase never touches the GPU at all.
EVAL_GPU_WORKERS = 8
EVAL_CPU_WORKERS = os.cpu_count() or 32

QAT_LR = 1e-4  # 10x below train.py's default 1e-3 -- fine-tuning from an
               # already-converged checkpoint, not training from scratch.

MAX_GRAD_NORM = 5.0  # gradient-norm clip for cluster annealing/QAT stages only
                      # (see train_one_epoch_clipped's docstring) -- NOT applied
                      # to resnet50's from-scratch float32 baseline training.

COARSE_FINE_THRESHOLD = 16  # num_clusters >= this: "coarse" (--anneal-epochs
                            # budget, constant LR). Below it: "fine" (bigger
                            # --qat-epochs budget, real milestone LR decay) --
                            # matches where every instability/collapse this
                            # pipeline has hit was actually observed.

ANNEAL_EPOCHS = 32  # fine-tune budget for each COARSE level (num_clusters
                    # >= COARSE_FINE_THRESHOLD) -- these sit close enough to
                    # full precision that a shorter budget at constant LR is
                    # normally plenty; per-level early stopping usually exits
                    # well before this anyway.

# Halving down to 16 (256,128,64,32,16), a coarser 16->12->8 step, then
# single-cluster steps below 8 (7,6,...,3) before the final target
# (--num-clusters, default 2). Every collapse/instability actually observed
# (both the unclipped-gradient NaN divergence and, even with clipping, the
# largest quality drops) happened in the <16 range specifically -- fine
# single-cluster steps there give training the gentlest possible landing at
# each drop. --schedule-stop-drop (see the main per-level loop) means the
# walk stops as soon as quality actually falls off, so this list is a CEILING
# on how far it explores, not a promise every level gets tried.
DEFAULT_ANNEAL_SCHEDULE = [256, 128, 64, 32, 16, 12, 8] + list(range(7, 2, -1))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--network", required=True, choices=[
        "mobilenetv1", "mobilenetv1_0.25", "mobilenetv1_0.50",
        "mobilenetv2", "resnet50", "resnet34", "resnet18",
    ])
    p.add_argument("--pretrained", default=None, help="Path (relative to this repo's own dir) to a float32 .pth")
    p.add_argument("--results-dir", required=True)
    p.add_argument("--qat-epochs", type=int, default=64,
                    help="Fine-tune budget for each FINE level (num_clusters < COARSE_FINE_THRESHOLD, "
                         "i.e. 12 and below in the default schedule) -- bigger than --anneal-epochs "
                         "since this is the range that actually needs to train through real "
                         "quantization error, not just a brief nudge; per-level early stopping usually "
                         "exits well before this.")
    p.add_argument("--baseline-epochs", type=int, default=128,
                    help="Only used when --pretrained is omitted (resnet50); matches this repo's "
                         "own from-scratch training scale (cfg_re50's own epochs=100) rather than a "
                         "shortcut, since there's no released checkpoint to start resnet50 from -- "
                         "profiled at ~7.5 min/epoch on this GPU, so 128 is ~16h")
    p.add_argument("--batch-size", type=int, default=None, help="Defaults to config.py's per-backbone value")
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--num-clusters", type=int, default=2)
    p.add_argument("--granularity", default="output", choices=["global", "output", "input"])
    p.add_argument("--anneal-schedule", type=int, nargs="*", default=DEFAULT_ANNEAL_SCHEDULE,
                    help="Cluster counts to pass through BEFORE --num-clusters, largest first "
                         "(fine-tuned --anneal-epochs at each, with the same early-stop-vs-baseline "
                         "check as the final target level) -- a one-shot jump straight to a small "
                         "codebook (num_clusters=2 in particular) was confirmed to collapse training "
                         "outright, and even gradual halving (16->8->4->2) still saw real quality "
                         "drops, so the default steps by 1 below 16 instead of halving. Values "
                         "<= --num-clusters are dropped automatically. Pass --anneal-schedule (empty) "
                         "to disable and go back to a one-shot jump.")
    p.add_argument("--anneal-epochs", type=int, default=ANNEAL_EPOCHS)
    p.add_argument("--schedule-stop-drop", type=float, default=0.01,
                    help="Stop walking DOWN the cluster schedule (not just the current level's own "
                         "epoch budget) as soon as a level's best quick-probe AP falls more than this "
                         "FRACTION below the float32 baseline's quick-probe AP (default 0.01 = 1%% "
                         "relative) -- once quality has actually started falling off, trying even "
                         "smaller cluster counts is very unlikely to recover, so this saves the "
                         "remaining schedule's training cost. The best level already reached is still "
                         "kept (see pick_best_level) even though the walk stopped early. Pass a large "
                         "value (e.g. 1e9) to disable and always walk the full --anneal-schedule.")
    p.add_argument("--full-eval-sample-size", type=int, default=None,
                    help="Debug/smoke-test only: caps the 'full' val-set eval too "
                         "(real sweep runs must omit this -- it makes the reported AP fake)")
    p.add_argument("--fused-bn", action="store_true",
                    help="Opt-in: quantize via quantize_graph, which fuses every Conv2d/BatchNorm2d "
                         "pair into a single quantized layer before quantizing (see "
                         "build_quantized_model's docstring). Default is OFF -- a resnet18 head-to-head "
                         "found this fused approach measurably WORSE than the legacy no-fuse "
                         "apply_lut_quantization path at every cluster level tried (e.g. selected level "
                         "0.1111 fused vs. 0.1116 no-fuse quick-probe AP, similar compression either "
                         "way), so no-fuse is the default; --fused-bn is kept only for further "
                         "experimentation, not because it currently wins.")
    return p.parse_args()


def build_data_loader(cfg, batch_size, num_workers):
    dataset = WiderFaceDetection(TRAIN_DATA, Augmentation(cfg["image_size"], RGB_MEAN))
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        collate_fn=dataset.collate_fn, pin_memory=True, drop_last=True,
    )


def build_criterion(cfg, device):
    priorbox = PriorBox(cfg, image_size=(cfg["image_size"], cfg["image_size"]))
    priors = priorbox.generate_anchors().to(device)
    return MultiBoxLoss(priors=priors, threshold=0.35, neg_pos_ratio=7, variance=cfg["variance"], device=device)


def train_one_epoch_clipped(model, criterion, optimizer, data_loader, epoch, device,
                            cfg, print_freq, max_grad_norm):
    """Same per-batch logic as this repo's own train.py's train_one_epoch
    (forward -> weighted loss -> zero_grad -> backward -> step, identical
    loss weighting/optimizer/logging), plus one addition:
    clip_grad_norm_ before optimizer.step(). train.py's own loop has no
    clipping at all -- fine for its own from-scratch/fine-tune training,
    but confirmed NOT fine for cluster annealing here: re-estimating a
    layer's LUT (anneal_lut_clusters) can shift a weight from its old
    center by a large relative amount all at once, and the very first
    epoch after a re-estimation (mobilenetv1_0.25, num_clusters 8->4 or
    4->2 in testing) produced NaN losses within the first ~50 batches with
    unclipped gradients. Kept as this driver's own loop rather than adding
    clipping to train.py itself, so that repo's own standard pipeline
    (used as-is for the resnet50 float32 baseline, where this instability
    doesn't apply) stays untouched.
    """
    model.train()
    batch_loss = []
    for batch_idx, (images, targets) in enumerate(data_loader):
        start_time = time.time()
        images = images.to(device)
        targets = [target.to(device) for target in targets]

        outputs = model(images)
        loss_loc, loss_conf, loss_land = criterion(outputs, targets)
        loss = cfg["loc_weight"] * loss_loc + loss_conf + loss_land

        # clip_grad_norm_ only guards the BACKWARD pass (exploding
        # gradients); it does nothing for a loss that's already non-finite
        # BEFORE backward() even runs (confirmed: mobilenetv1_0.25,
        # num_clusters=12, epoch 25 -- a single pathological batch produced
        # an inf forward loss, likely an exp() overflow in the box-decode
        # math MultiBoxLoss uses internally; backward() on an inf loss
        # produces NaN gradients, and NaN * any clip scale factor is still
        # NaN, so the optimizer step corrupted every parameter to NaN in
        # one shot -- confirmed the training process was dead immediately
        # after). Skipping the step entirely for a non-finite loss (instead
        # of clipping something that can't be clipped) is the actual fix --
        # one bad batch just contributes nothing instead of destroying the
        # whole run.
        if not torch.isfinite(loss):
            print(f"[train_lut_quant] WARNING: non-finite loss ({loss.item()}) at "
                  f"epoch {epoch + 1} batch {batch_idx + 1} -- skipping this batch's update")
            optimizer.zero_grad()
            continue

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        if (batch_idx + 1) % print_freq == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch: {epoch + 1}/{cfg['epochs']} | Batch: {batch_idx + 1}/{len(data_loader)} | "
                f"Loss Localization : {loss_loc.item():.4f} | Classification: {loss_conf.item():.4f} | "
                f"Landmarks: {loss_land.item():.4f} | "
                f"LR: {lr:.8f} | Time: {(time.time() - start_time):.4f} s"
            )
        batch_loss.append(loss.item())
    print(f"Average batch loss: {sum(batch_loss) / len(batch_loss):.7f}")


def run_training(model, cfg, device, epochs, lr, milestones, batch_size, num_workers,
                 print_freq=50, on_epoch_end=None, max_grad_norm=None):
    """Thin wrapper around this repo's own train.py's train_one_epoch --
    same optimizer type/momentum/weight-decay/loss-weighting/augmentation
    as its standard pipeline, just with epochs/lr/milestones as explicit
    parameters (train.py hardcodes these to config.py's from-scratch
    values) and an optional end-of-epoch hook for the QAT early-stop probe.

    max_grad_norm: when given, uses train_one_epoch_clipped (this file's
    own loop, with gradient clipping) instead of train.py's own
    train_one_epoch -- see that function's docstring for why cluster
    annealing needs it and from-scratch/baseline training doesn't.
    """
    # train_one_epoch reads cfg as a module global, including cfg['epochs']
    # purely for its own progress-print denominator -- copy so that print
    # reflects the ACTUAL budget passed here (e.g. QAT's reduced epochs),
    # not whatever this cfg dict's from-scratch value happened to be.
    cfg = dict(cfg)
    cfg["epochs"] = epochs
    retina_train.cfg = cfg
    data_loader = build_data_loader(cfg, batch_size, num_workers)
    criterion = build_criterion(cfg, device)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

    for epoch in range(epochs):
        if max_grad_norm is not None:
            train_one_epoch_clipped(model, criterion, optimizer, data_loader, epoch, device,
                                    cfg, print_freq, max_grad_norm)
        else:
            retina_train.train_one_epoch(model, criterion, optimizer, data_loader, epoch, device, print_freq)
        lr_scheduler.step()
        if on_epoch_end is not None:
            stop = on_epoch_end(epoch)
            if stop:
                break


def scaled_milestones(cfg, epochs):
    orig_epochs = cfg["epochs"]
    return sorted(set(max(1, round(m / orig_epochs * epochs)) for m in cfg["milestones"]))


def build_quantized_model(cfg, device, num_clusters, granularity, fused_bn=False):
    """Fresh RetinaFace, LUT-quantized at exactly num_clusters -- used to
    reconstruct the right architecture for loading a specific cluster
    level's saved checkpoint (each level's weight_fake_quant.lut buffer has
    a different shape, so a checkpoint from one level can't be
    strict-loaded into a model quantized at another).

    Default (fused_bn=False) uses apply_lut_quantization -- BatchNorm2d
    stays a separate module, and only the Conv2d weight itself gets
    quantized. This is the scheme that actually ships: a resnet18
    head-to-head against quantize_graph's fuse-then-quantize (below) found
    apply_lut_quantization measurably BETTER at every cluster level tried
    (e.g. selected level 0.1116 vs. 0.1111 quick-probe AP, essentially
    identical compression ratio either way).

    fused_bn=True instead uses quantize_graph (train/refine/quant_utils.py),
    which fuses every Conv2d/BatchNorm2d pair in the graph FIRST
    (fuse_conv_bn_graph) and only then quantizes the fused weight --
    BatchNorm2d no longer exists as a separate module afterward, its effect
    baked into the quantized Conv2d's own weight/bias, matching what a real
    fused-inference deployment ships. Kept as an opt-in for further
    experimentation, not because it currently wins.
    """
    model = RetinaFace(cfg=cfg).to(device)
    if fused_bn:
        quantize_graph(model, granularity=granularity, num_clusters=num_clusters, scheme="lut_kmeans")
    else:
        apply_lut_quantization(model, num_clusters=num_clusters, granularity=granularity)
    return model.to(device)


def pick_best_level(level_log, baseline_quick_map, quick_probe_tolerance: float = 0.005):
    """Among every cluster level actually reached, prefer the level with the
    HIGHEST measured compression_ratio among those whose quick-probe quality
    met/exceeded the float32 baseline (within quick_probe_tolerance -- see
    below); if none did, fall back to the best quality seen among levels
    that still compress at all (ratio > 1.0); if NOTHING compresses even
    loosely, fall back to the single best-quality level seen regardless of
    size -- "don't ship a broken model" beats "ship the smallest one" when
    nothing reached the quality bar.

    Deliberately does NOT use num_clusters as a stand-in for compression:
    at "output" (per-channel) granularity, a HIGH cluster count's own
    codebook (num_clusters floats PER OUTPUT CHANNEL) can outweigh the
    savings from fewer index bits entirely -- confirmed on
    mobilenetv1_0.25, where the num_clusters=256 level (selected purely on
    quality before this fix) measured compression_ratio=0.43x, i.e.
    LARGER than float32, not smaller. compression_ratio here is always the
    real, measured number (see estimate_compression_ratio), never inferred
    from cluster count.

    quick_probe_tolerance guards against exactly the failure this was
    confirmed to have on mobilenetv1_0.25: a strict `>=` against the raw
    quick-probe (400-image, inherently noisy) baseline let k=12 -- whose
    REAL full-val AP (0.8531) and compression (4.24x) both beat k=16's
    (0.8509, 3.73x) -- get excluded from "qualifying" for missing the
    threshold by 0.0000113 (~0.01% relative), pure sampling noise, handing
    the pick to a level that was actually worse on every real axis. A small
    relative tolerance (default 0.5%, comfortably larger than that miss but
    far smaller than the ~1%+ drops that should still disqualify a level)
    treats the quick-probe as the noisy estimate it is instead of a
    razor's-edge gate.
    """
    compressing = [lvl for lvl in level_log if lvl["compression_ratio"] > 1.0]
    threshold = baseline_quick_map * (1.0 - quick_probe_tolerance)
    qualifying = [lvl for lvl in compressing if lvl["mean_ap_quick"] >= threshold]
    if qualifying:
        return max(qualifying, key=lambda lvl: lvl["compression_ratio"])
    if compressing:
        return max(compressing, key=lambda lvl: lvl["mean_ap_quick"])
    return max(level_log, key=lambda lvl: lvl["mean_ap_quick"])


def evaluate(model, cfg, device, pred_dir, sample_size=None, sample_seed=QUICK_PROBE_SEED):
    was_training = model.training
    model.eval()
    # Training always feeds a FIXED 640x640 crop (Augmentation), so the CUDA
    # caching allocator only ever needs one activation-tensor shape per
    # layer all epoch; WIDER FACE eval images each resize to their OWN
    # aspect-preserving (height, width), a different shape practically every
    # single image. Confirmed (isolated timing) that reusing the allocator's
    # state straight from a just-finished training epoch, unchanged, turns
    # what's an ~85s pass on a fresh model into 10+ minutes here -- each
    # new, never-before-seen shape apparently can't be served from the
    # blocks training's alloc/free pattern left cached, forcing far more
    # expensive underlying cudaMalloc traffic than a clean allocator would
    # need. Releasing everything training left cached (nothing here is
    # still in use -- optimizer/gradients from that epoch are done) lets
    # eval's first-time shapes allocate from a clean slate instead.
    torch.cuda.empty_cache()
    aps = we.evaluate_model(
        model, cfg, device, DATASET_FOLDER, VAL_LIST, GT_DIR, pred_dir,
        sample_size=sample_size, sample_seed=sample_seed,
        num_workers=EVAL_GPU_WORKERS, eval_num_workers=EVAL_CPU_WORKERS,
    )
    if was_training:
        model.train()
    return aps


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    report = {"network": args.network, "num_clusters": args.num_clusters, "granularity": args.granularity}

    cfg = dict(get_config(args.network))
    batch_size = args.batch_size or cfg["batch_size"]

    model = RetinaFace(cfg=cfg).to(device)

    t0 = time.time()
    baseline_path = results_dir / f"{args.network}_baseline_fp32.pth"
    if args.pretrained:
        pretrained_path = RETINA_DIR / args.pretrained
        state_dict = torch.load(pretrained_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        report["baseline_source"] = str(pretrained_path)
        report["baseline_trained_epochs"] = 0
        print(f"[{args.network}] loaded pretrained checkpoint from {pretrained_path}")
    elif baseline_path.exists():
        # A prior run for this exact results-dir may have already trained and
        # saved the float32 baseline (e.g. this process got killed partway
        # through a LATER level) -- reload it instead of redoing a 128-epoch
        # (~16h) baseline train from scratch. Mirrors the per-level resume
        # check below.
        print(f"[{args.network}] baseline checkpoint already exists at {baseline_path} -- "
              f"resuming from it instead of retraining")
        model.load_state_dict(torch.load(baseline_path, map_location=device, weights_only=True))
        report["baseline_source"] = str(baseline_path)
        report["baseline_trained_epochs"] = 0
    else:
        print(f"[{args.network}] no pretrained checkpoint -- training a "
              f"{args.baseline_epochs}-epoch float32 baseline from scratch")
        milestones = scaled_milestones(cfg, args.baseline_epochs)
        run_training(model, cfg, device, args.baseline_epochs, lr=1e-3, milestones=milestones,
                     batch_size=batch_size, num_workers=args.num_workers)
        torch.save(model.state_dict(), baseline_path)
        report["baseline_source"] = "trained_from_scratch"
        report["baseline_trained_epochs"] = args.baseline_epochs
        print(f"[{args.network}] baseline training done in {time.time() - t0:.1f}s -> {baseline_path}")

    # --- float32 baseline quality: real val-set AP (reported number) + a
    # fixed-sample quick probe (early-stop reference for QAT below) ---
    t0 = time.time()
    baseline_aps_full = evaluate(model, cfg, device, str(results_dir / "pred_baseline_full"),
                                 sample_size=args.full_eval_sample_size)
    report["baseline_aps_full"] = baseline_aps_full
    print(f"[{args.network}] baseline full-val AP: {baseline_aps_full} "
          f"({time.time() - t0:.1f}s)")

    baseline_aps_quick = evaluate(model, cfg, device, str(results_dir / "pred_baseline_quick"),
                                  sample_size=QUICK_PROBE_SIZE)
    baseline_quick_map = we.mean_ap(baseline_aps_quick)
    report["baseline_aps_quick"] = baseline_aps_quick
    print(f"[{args.network}] baseline quick-probe mean AP (early-stop target): {baseline_quick_map:.4f}")

    # --- gradual cluster annealing: quantize to the largest (gentlest)
    # schedule step first, fine-tune with early stopping against the
    # float32 baseline quality, re-estimate the LUT at the next SMALLER
    # count from the now-adapted weights, repeat down to the final target
    # num_clusters -- EVERY level gets the same early-stop-vs-baseline
    # treatment and its own saved checkpoint, not just the last one. A
    # one-shot jump straight to a small codebook (num_clusters=2 in
    # particular) was confirmed to collapse training outright; see
    # anneal_lut_clusters's docstring. Tracking every level's own quality
    # means the final pick doesn't depend on the smallest target actually
    # working -- pick_best_level below falls back to whichever level
    # actually held quality if the aggressive end of the schedule doesn't
    # recover in the epoch budget given.
    #
    # Default (--fused-bn not passed) uses apply_lut_quantization: BatchNorm2d
    # stays a separate module, only the Conv2d weight gets quantized -- see
    # build_quantized_model's docstring for the resnet18 head-to-head that
    # found this measurably BETTER than fusing-then-quantizing. --fused-bn
    # opts into quantize_graph instead, which fuses every Conv2d/BatchNorm2d
    # pair (fuse_conv_bn_graph) before quantizing the fused weight --
    # BatchNorm2d no longer exists as a separate module in `model` after that
    # call, its effect folded into each fused layer's own quantized
    # weight/bias.
    anneal_steps = sorted({k for k in args.anneal_schedule if k > args.num_clusters}, reverse=True)
    cluster_schedule = anneal_steps + [args.num_clusters]
    report["cluster_schedule"] = cluster_schedule
    print(f"[{args.network}] cluster annealing schedule: {cluster_schedule}")

    if args.fused_bn:
        quantize_graph(model, granularity=args.granularity, num_clusters=cluster_schedule[0], scheme="lut_kmeans")
    else:
        apply_lut_quantization(model, num_clusters=cluster_schedule[0], granularity=args.granularity)
    model.to(device)

    level_log = []
    for step_idx, k in enumerate(cluster_schedule):
        is_last = step_idx == len(cluster_schedule) - 1
        # Coarse levels (num_clusters >= COARSE_FINE_THRESHOLD, i.e. close
        # to full precision) get the lighter --anneal-epochs budget at
        # constant LR; fine levels (< threshold, the range every observed
        # instability/collapse actually happened in) get the bigger
        # --qat-epochs budget AND real (scaled) milestone LR decay -- a
        # short constant-LR nudge was enough near-losslessly at k=256, but
        # not for a level that has real quantization error to train
        # through. Per-level early stopping (below) still exits the moment
        # a level matches baseline, so most levels use far less than their
        # budget in practice -- this budget only matters for levels that
        # actually need it.
        is_fine = k < COARSE_FINE_THRESHOLD
        epochs = args.qat_epochs if is_fine else args.anneal_epochs
        milestones = scaled_milestones(cfg, epochs) if is_fine else []

        checkpoint_path = results_dir / f"{args.network}_lut{k}_checkpoint.pth"

        # Per-level resume: a prior run for this exact results-dir may have
        # already trained and saved this level's checkpoint (e.g. this
        # process got killed partway through a LATER level) -- reload it and
        # skip straight to this level's evaluation/logging instead of
        # retraining from scratch, which is what happened before this check
        # existed (confirmed wasteful directly: killed mid-epoch-4/64 of
        # num_clusters=12 with 256/128/64/32/16 all already fully trained and
        # checkpointed on disk).
        if checkpoint_path.exists():
            print(f"[{args.network}] num_clusters={k} checkpoint already exists at {checkpoint_path} -- "
                  f"resuming from it instead of retraining")
            model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
            aps = evaluate(model, cfg, device, str(results_dir / f"pred_level_k{k}"), sample_size=QUICK_PROBE_SIZE)
            best_quick_map = we.mean_ap(aps)
            epoch_log = [{"epoch": -1, "aps_quick": aps, "mean_ap_quick": best_quick_map, "resumed": True}]
            level_time = 0.0
        else:
            best_state = copy.deepcopy(model.state_dict())
            best_quick_map = -1.0
            epoch_log = []

            def on_epoch_end(epoch, k=k, epochs=epochs):
                nonlocal best_state, best_quick_map
                aps = evaluate(model, cfg, device, str(results_dir / f"pred_level_k{k}"), sample_size=QUICK_PROBE_SIZE)
                quick_map = we.mean_ap(aps)
                epoch_log.append({"epoch": epoch, "aps_quick": aps, "mean_ap_quick": quick_map})
                print(f"[{args.network}] num_clusters={k} epoch {epoch + 1}/{epochs} quick-probe mean AP: "
                      f"{quick_map:.4f} (target {baseline_quick_map:.4f})")
                if quick_map > best_quick_map:
                    best_quick_map = quick_map
                    best_state = copy.deepcopy(model.state_dict())
                if quick_map >= baseline_quick_map:
                    print(f"[{args.network}] num_clusters={k} reached/exceeded float32 quick-probe quality "
                          f"-- stopping this level early")
                    return True
                return False

            t0 = time.time()
            run_training(model, cfg, device, epochs, lr=QAT_LR, milestones=milestones,
                         batch_size=batch_size, num_workers=args.num_workers, on_epoch_end=on_epoch_end,
                         max_grad_norm=MAX_GRAD_NORM)
            level_time = time.time() - t0

            model.load_state_dict(best_state)
            torch.save(model.state_dict(), checkpoint_path)

        # Measured, not inferred from num_clusters: at "output" (per-channel)
        # granularity, a high cluster count's own codebook (num_clusters
        # floats PER OUTPUT CHANNEL) can outweigh the index-bit savings
        # entirely, so this is the ONLY reliable signal pick_best_level can
        # use to tell a level that actually shrinks the model from one that
        # doesn't -- see that function's docstring for a concrete case
        # (num_clusters=256 measured 0.43x, i.e. LARGER than float32).
        level_compression = estimate_compression_ratio(model)

        level_log.append({
            "num_clusters": k, "epochs_run": len(epoch_log), "epochs_budget": epochs,
            "mean_ap_quick": best_quick_map, "epoch_log": epoch_log,
            "time_seconds": level_time, "checkpoint_path": str(checkpoint_path),
            "early_stopped": len(epoch_log) < epochs,
            "compression_ratio": level_compression["compression_ratio"],
            "compressed_mb": level_compression["compressed_mb"],
        })
        print(f"[{args.network}] num_clusters={k} done in {level_time:.1f}s, "
              f"best quick-probe mean AP: {best_quick_map:.4f}, "
              f"compression ratio: {level_compression['compression_ratio']:.2f}x -> {checkpoint_path}")

        # Stop walking the WHOLE schedule (not just this level's own epoch
        # budget) once quality has actually fallen off by more than
        # --schedule-stop-drop -- trying an even smaller cluster count from
        # here essentially never recovers (confirmed: quality only ever
        # gets worse continuing past the point it first drops), so this
        # saves the remaining levels' training cost. pick_best_level still
        # picks the best level reached so far; nothing about the selection
        # logic changes, only how far the walk goes before selecting.
        relative_drop = (baseline_quick_map - best_quick_map) / baseline_quick_map if baseline_quick_map else 0.0
        if not is_last and relative_drop > args.schedule_stop_drop:
            print(f"[{args.network}] num_clusters={k} quality dropped {relative_drop:.1%} below baseline "
                  f"(> {args.schedule_stop_drop:.1%} threshold) -- stopping the cluster schedule here")
            break

        if not is_last:
            next_k = cluster_schedule[step_idx + 1]
            anneal_lut_clusters(model, num_clusters=next_k)
            print(f"[{args.network}] re-estimated LUT at num_clusters={next_k}")

    report["level_log"] = level_log

    # --- pick the level to actually report: smallest num_clusters that
    # matched/beat the float32 baseline, or the best-quality level seen if
    # none did (see pick_best_level's docstring) -- rebuild a model
    # quantized at exactly that level's cluster count (buffer shapes differ
    # per level) and load its saved checkpoint, since the live `model`
    # object above is left at the LAST schedule step, not necessarily the
    # selected one. ---
    selected = pick_best_level(level_log, baseline_quick_map)
    report["selected_level"] = {"num_clusters": selected["num_clusters"], "mean_ap_quick": selected["mean_ap_quick"]}
    print(f"[{args.network}] selected num_clusters={selected['num_clusters']} "
          f"(quick-probe mean AP {selected['mean_ap_quick']:.4f}) as the best quality/compression tradeoff")

    del model
    gc.collect()
    torch.cuda.empty_cache()

    selected_model = build_quantized_model(cfg, device, selected["num_clusters"], args.granularity, fused_bn=args.fused_bn)
    selected_model.load_state_dict(torch.load(selected["checkpoint_path"], map_location=device, weights_only=True))

    compression = estimate_compression_ratio(selected_model)
    report["compression"] = compression
    print(f"[{args.network}] selected-level compression ratio: {compression['compression_ratio']:.2f}x "
          f"({compression['float32_mb']:.2f}MB -> {compression['compressed_mb']:.2f}MB)")

    # --- selected level: real val-set AP (the actually-reported number) ---
    final_aps_full = evaluate(selected_model, cfg, device, str(results_dir / "pred_final_full"),
                              sample_size=args.full_eval_sample_size)
    report["final_aps_full"] = final_aps_full
    print(f"[{args.network}] FINAL (num_clusters={selected['num_clusters']}) quantized full-val AP: {final_aps_full}")

    torch.save(selected_model.state_dict(), results_dir / f"{args.network}_lut{selected['num_clusters']}_final.pth")
    with open(results_dir / "report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"[{args.network}] done -> {results_dir / 'report.json'}")

    del selected_model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
