"""Runs ONE real (unsampled, all 3226 images) WIDER FACE full-val AP pass for
ONE checkpoint, as a standalone subprocess -- the unit of work run_parallel.py
launches N of concurrently.

Why a subprocess boundary and not a thread/async/in-process loop: each
invocation gets its own fresh Python process (own CUDA context, own GIL),
so N of these can run truly in parallel on this box's many CPU cores and
mostly-idle GPU. This sidesteps the exact trap documented in
widerface_eval.py's module docstring -- that a ProcessPoolExecutor
splitting the PER-IMAGE work of a single eval regressed (fork-after-CUDA-init
hazard, and each task pickling a full decoded image array across a pipe).
Here there is no cross-process data transfer at all: each subprocess reads
its own checkpoint from disk, runs its own complete eval loop, and writes
only a small JSON summary at the end -- nothing pickled, nothing shared.

One eval (float32 mobilenetv1_0.25, full 3226 images) measured ~576s serial
in this session; GPU utilization during that pass was ~1-12% (the NMS/decode
postprocessing is what's slow, not the forward pass -- see widerface_eval.py),
so running several of these concurrently is expected to shorten wall-clock
for a whole-schedule report roughly by the concurrency factor, not slow any
individual pass down much, as long as concurrency stays well under this
box's CPU core count and the small per-context GPU memory overhead stays
under the card's free memory.

IMPORTANT: torch defaults to using EVERY core for its own intra-op thread
pool unless told otherwise -- confirmed directly (128 cores, unset
OMP_NUM_THREADS/MKL_NUM_THREADS -> each of N concurrent workers tries to
claim all 128). Running many of these concurrently without capping that
oversubscribes the CPU by a factor of N, which starves the very CPU-bound
decode/NMS work this whole parallel-across-levels design depends on --
observed directly as workers producing almost no predictions and GPU
utilization reading ~0% (not because the GPU had nothing to do, but because
the CPU-side work feeding it was stuck in thread-scheduling contention).
--num-threads (set via torch.set_num_threads below, expected to be set by
the caller to roughly cores / concurrency) is the fix; run_parallel.py
also exports OMP_NUM_THREADS/MKL_NUM_THREADS into each subprocess's own
environment as a second guarantee, since some native libraries only read
those at process start rather than respecting torch.set_num_threads.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

_RETINA_DIR = Path(__file__).resolve().parent.parent
_REPO_ROOT = Path("/workspace/home_0/work/llwll")

if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from train.refine.quant_utils import quantize_graph, apply_lut_quantization  # noqa: E402

if str(_RETINA_DIR) not in sys.path:
    sys.path.insert(0, str(_RETINA_DIR))
from config import get_config  # noqa: E402
from models import RetinaFace  # noqa: E402
import widerface_eval as we  # noqa: E402

DATASET_FOLDER = str(_RETINA_DIR / "data/widerface/val/images/")
VAL_LIST = str(_RETINA_DIR / "data/widerface/val/wider_val.txt")
GT_DIR = str(_RETINA_DIR / "widerface_evaluation/ground_truth")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--network", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--num-clusters", type=int, default=None,
                    help="omit for a plain float32 checkpoint")
    p.add_argument("--granularity", default="output", choices=["global", "output", "input"])
    p.add_argument("--label", required=True, help="row label for the aggregated report, e.g. 'k=16'")
    p.add_argument("--out", required=True, help="path to write this level's result JSON")
    p.add_argument("--pred-dir", required=True)
    p.add_argument("--num-workers", type=int, default=2,
                    help="ThreadPoolExecutor width for widerface_eval.py's per-image loop "
                         "(within THIS worker's own eval, separate from how many of these "
                         "worker processes run.py launches concurrently)")
    p.add_argument("--log-every", type=int, default=10,
                    help="log a memory snapshot every N completed images (independent of "
                         "training batch size, which is otherwise the chunk-submission size)")
    p.add_argument("--num-threads", type=int, default=4,
                    help="cap on torch's own intra-op CPU thread pool for THIS process -- "
                         "torch defaults to using every core, which oversubscribes badly when "
                         "many of these run concurrently (see module docstring)")
    p.add_argument("--device", default=None, choices=["cuda", "cpu"],
                    help="force a device instead of auto-picking cuda when available -- 'cpu' "
                         "lets a full-val batch run entirely off the GPU while something else "
                         "(e.g. a training run) needs it uncontended. Eval wall time here is "
                         "already dominated by CPU-bound NMS/decode, not the forward pass (see "
                         "widerface_eval.py), so CPU-only isn't as large a slowdown as it would "
                         "be for training.")
    p.add_argument("--fused-bn", action="store_true",
                    help="Must match how the checkpoint being loaded was quantized (see "
                         "train_lut_quant.py's build_quantized_model docstring) -- default is the "
                         "no-fuse apply_lut_quantization scheme (BatchNorm2d stays separate), which "
                         "is also train_lut_quant.py's default. Pass this only when evaluating a "
                         "checkpoint that was itself trained with --fused-bn (quantize_graph); loading "
                         "a fused checkpoint into a no-fuse-reconstructed model (or vice versa) fails "
                         "with a state_dict key mismatch since fusion removes the separate BatchNorm2d "
                         "module entirely.")
    args = p.parse_args()

    torch.set_num_threads(args.num_threads)

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = dict(get_config(args.network))

    model = RetinaFace(cfg=cfg).to(device)
    if args.num_clusters is not None:
        if args.fused_bn:
            quantize_graph(model, granularity=args.granularity, num_clusters=args.num_clusters, scheme="lut_kmeans")
        else:
            apply_lut_quantization(model, num_clusters=args.num_clusters, granularity=args.granularity)
        model.to(device)
    state_dict = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    t0 = time.time()
    aps = we.evaluate_model(model, cfg, device, DATASET_FOLDER, VAL_LIST, GT_DIR,
                            pred_dir=args.pred_dir, sample_size=None,
                            num_workers=args.num_workers, log_progress=True, log_every=args.log_every,
                            eval_num_workers=args.num_threads)
    elapsed = time.time() - t0

    result = {
        "label": args.label,
        "num_clusters": args.num_clusters,
        "checkpoint": args.checkpoint,
        "aps_full": aps,
        "mean_ap_full": we.mean_ap(aps),
        "time_seconds": elapsed,
    }
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(f"[worker] {args.label}: easy={aps['easy']:.4f} medium={aps['medium']:.4f} "
          f"hard={aps['hard']:.4f} ({elapsed:.1f}s) -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
