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
_INFER_DIR = _RETINA_DIR / "inference"

if str(_RETINA_DIR) not in sys.path:
    sys.path.insert(0, str(_RETINA_DIR))
if str(_INFER_DIR) not in sys.path:
    sys.path.insert(0, str(_INFER_DIR))
from config import get_config  # noqa: E402
import widerface_eval_mp as we  # noqa: E402
from export_common import load_plain_with_clusters, cluster_count  # noqa: E402

DATASET_FOLDER = str(_RETINA_DIR / "data/widerface/val/images/")
VAL_LIST = str(_RETINA_DIR / "data/widerface/val/wider_val.txt")
GT_DIR = str(_RETINA_DIR / "widerface_evaluation/ground_truth")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--network", required=True)
    p.add_argument("--checkpoint", required=True,
                    help="a plain+clusters checkpoint -- EITHER a combined .zip "
                         "(export_pytorch_batch_hf.py's packing, {state_dict, clusters} in one "
                         "file) or a bare .pth (plain float32 state_dict, no clusters) -- see "
                         "inference/export_common.py's load_plain_with_clusters. Never a raw "
                         "training checkpoint: this repo's inference/eval scripts build only "
                         "from that plain-weights-plus-cluster-info source, never this "
                         "project's private training machinery.")
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
    p.add_argument("--device", default=None, choices=["cuda", "mps", "cpu"],
                    help="force a device instead of auto-picking the best available accelerator "
                         "(cuda, then Apple Silicon mps, then cpu) -- 'cpu' "
                         "lets a full-val batch run entirely off the GPU while something else "
                         "(e.g. a training run) needs it uncontended. Eval wall time here is "
                         "already dominated by CPU-bound NMS/decode, not the forward pass (see "
                         "widerface_eval.py), so CPU-only isn't as large a slowdown as it would "
                         "be for training.")
    args = p.parse_args()

    torch.set_num_threads(args.num_threads)

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = dict(get_config(args.network))

    # load_plain_with_clusters needs no reconstruction step (no quantize_graph/
    # apply_lut_quantization) -- the checkpoint's weights are already plain
    # floats snapped to their trained cluster values, so a stock RetinaFace's
    # load_state_dict() is all that's needed regardless of num_clusters.
    model, cluster_info = load_plain_with_clusters(cfg, args.checkpoint)
    model = model.to(device)
    num_clusters = cluster_count(cluster_info)

    t0 = time.time()
    aps = we.evaluate_model(model, cfg, device, DATASET_FOLDER, VAL_LIST, GT_DIR,
                            pred_dir=args.pred_dir, sample_size=None,
                            num_workers=args.num_workers, log_progress=True, log_every=args.log_every,
                            eval_num_workers=args.num_threads)
    elapsed = time.time() - t0

    result = {
        "label": args.label,
        "num_clusters": num_clusters,
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
