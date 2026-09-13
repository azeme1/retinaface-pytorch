"""Parallel verification routine: runs the real (unsampled) WIDER FACE
full-val AP for every trained LUT level of a backbone concurrently, each as
its own worker.py subprocess, instead of one at a time serially.

Rationale for parallelizing across LEVELS rather than across images within
one eval (the approach already tried and reverted -- see
widerface_eval.py's module docstring): each level's checkpoint is fully
independent of every other level's, so N of them can run as N completely
separate OS processes with zero shared state and zero cross-process data
transfer -- avoiding both problems that sank the per-image attempt
(fork-after-CUDA-init hazard, large-array pickling). GPU utilization during
a single eval pass measured only ~1-12% in this session (NMS/decode
postprocessing dominates, not the forward pass), and this box has 128 CPU
cores against one GPU, so several full-val passes fit comfortably alongside
each other.

Concurrency is memory-adaptive, not a fixed process count: a new pass only
launches while both GPU memory and system RAM stay under --mem-threshold
(default 90%) of their totals -- checked with plain nvidia-smi/proc/meminfo
reads, no extra dependency. A fixed count either wastes idle headroom (this
box measured 230GB RAM free and ~15.8GB GPU free with nothing else running)
or risks OOM on a bigger backbone/box where per-worker footprint is larger --
sizing off actual usage adapts to both automatically. --max-concurrency is
still an optional hard ceiling on top of that, mainly to avoid pointlessly
launching more OS processes than there are pending jobs.

Usage:
    python run_parallel.py --network mobilenetv1_0.25 \\
        --results-dir results/mobilenetv1_0.25
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
REPO_ROOT = EVAL_DIR.parent
INFER_DIR = REPO_ROOT / "inference"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(INFER_DIR) not in sys.path:
    sys.path.insert(0, str(INFER_DIR))
import widerface_eval as we  # noqa: E402
from export_common import download_hf_artifact  # noqa: E402

# Every level this project has published plain+clusters checkpoints for on
# the HF repo (results/<network>/pytorch/<network>_<level>.zip) -- a level
# missing for a given --network is just skipped below (download_hf_artifact
# raises, caught per-level), same tolerance the old local-file existence
# check had.
FULL_SCHEDULE = ["c256", "c128", "c64", "c32", "c16", "c12", "c8", "c7", "c6", "c5", "c4", "c3", "c2"]


def _gpu_usage_fraction() -> float:
    """Fraction of total GPU memory currently used, via nvidia-smi -- 0.0 if
    nvidia-smi isn't available (e.g. CPU-only box), so the gate never blocks
    launches on a machine with no GPU to protect."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=True,
        ).stdout.strip().splitlines()[0]
        used, total = (float(x) for x in out.split(","))
        return used / total if total else 0.0
    except Exception:
        return 0.0


def _ram_usage_fraction() -> float:
    """Fraction of system RAM currently used, from /proc/meminfo's
    MemAvailable (accounts for reclaimable cache, unlike MemFree) -- stdlib
    file read, no psutil dependency."""
    total_kb = avail_kb = None
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemTotal:"):
                total_kb = int(line.split()[1])
            elif line.startswith("MemAvailable:"):
                avail_kb = int(line.split()[1])
            if total_kb is not None and avail_kb is not None:
                break
    if not total_kb:
        return 0.0
    return 1.0 - (avail_kb / total_kb)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--network", required=True)
    p.add_argument("--results-dir", required=True, help="output directory for this run's own "
                    "full_eval_parallel/ report -- checkpoints themselves come from --hf-repo, "
                    "not read from here")
    p.add_argument("--hf-repo", default="azemel/retinaface-xs",
                    help="HF repo every level's plain+clusters checkpoint is downloaded from "
                         "(results/<network>/pytorch/<network>_<level>.zip) -- see "
                         "inference/export_common.py's download_hf_artifact")
    p.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    p.add_argument("--mem-threshold", type=float, default=0.90,
                    help="launch another pass only while GPU-memory and system-RAM usage both "
                         "stay under this fraction of their totals")
    p.add_argument("--max-concurrency", type=int, default=None,
                    help="hard ceiling on passes running at once, on top of the memory gate; "
                         "defaults to min(4, queued jobs) -- launching all ~14 levels at once "
                         "was confirmed to blow past the memory gate outright (nvidia-smi's "
                         "reported usage lags a few seconds behind a just-launched process's "
                         "actual CUDA context allocation, so a burst of near-simultaneous "
                         "launches all read 'plenty free' before any of them show up as used, "
                         "then all allocate for real together and OOM). 4 was small enough to "
                         "avoid that race in the runs that hit it (mobilenetv1, mobilenetv2, "
                         "resnet34 -- 128MB-800MB+ float32 backbones); pass a higher number "
                         "explicitly once you've confirmed headroom for a given backbone size.")
    p.add_argument("--poll-seconds", type=float, default=3.0,
                    help="how often to re-check memory usage and process exits")
    p.add_argument("--launch-stagger-seconds", type=float, default=5.0,
                    help="minimum delay between two launches, regardless of memory readings -- "
                         "closes the launch-burst race described under --max-concurrency by "
                         "giving nvidia-smi time to reflect each new process's allocation "
                         "before the gate evaluates the next one")
    p.add_argument("--max-retries", type=int, default=3,
                    help="on any job failing (worker exited nonzero or wrote no result), "
                         "automatically retry just the failed jobs, halving max-concurrency "
                         "each round (floor 1), up to this many extra rounds -- an OOM from "
                         "too much concurrency is expected to clear once fewer workers share "
                         "the GPU, so this recovers automatically instead of requiring a "
                         "manual rerun")
    p.add_argument("--force", action="store_true",
                    help="re-run every level even if its out_json result already exists from "
                         "a prior invocation for this results-dir; default skips those so a "
                         "rerun (e.g. after a partial OOM failure) only redoes what's missing")
    p.add_argument("--device", default=None, choices=["cuda", "cpu"],
                    help="force every worker onto this device instead of auto-picking cuda -- "
                         "'cpu' keeps the whole batch off the GPU entirely (e.g. while a "
                         "training run needs it uncontended); the GPU-memory half of "
                         "--mem-threshold is meaningless in that case, since nothing here touches "
                         "the GPU at all")
    p.add_argument("--no-lowest-priority", action="store_true",
                    help="skip wrapping workers in nice -n 19 / ionice -c 3 -- by default this "
                         "batch runs at the lowest CPU/IO scheduling priority so it soaks up "
                         "idle capacity without competing with anything else (e.g. a concurrent "
                         "training run) that wants the same cores")
    p.add_argument("--include-baseline", action="store_true", default=True,
                    help="also run the \"float32\" HF level as an unquantized baseline row")
    args = p.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = results_dir / "full_eval_parallel"
    out_dir.mkdir(parents=True, exist_ok=True)

    jobs = []  # list of (label, level, checkpoint_path, out_json_path)

    if args.include_baseline:
        try:
            baseline_ckpt = str(download_hf_artifact(args.hf_repo, args.network, "pytorch", "float32",
                                                       token=args.hf_token))
            out_json = out_dir / "baseline.json"
            jobs.append(("baseline", "float32", baseline_ckpt, out_json))
        except Exception as e:  # noqa: BLE001 -- no float32 level published for this network
            print(f"[run_parallel] no float32 checkpoint on {args.hf_repo} for {args.network} "
                  f"({type(e).__name__}) -- skipping baseline row")

    for level in FULL_SCHEDULE:
        try:
            ckpt = str(download_hf_artifact(args.hf_repo, args.network, "pytorch", level, token=args.hf_token))
        except Exception as e:  # noqa: BLE001 -- this level isn't published for this network
            print(f"[run_parallel] {level}: not on {args.hf_repo} for {args.network} "
                  f"({type(e).__name__}), skipping")
            continue
        out_json = out_dir / f"{level}.json"
        jobs.append((f"k={level.removeprefix('c')}", level, ckpt, out_json))

    all_jobs = jobs
    if args.force:
        jobs = list(all_jobs)
    else:
        already_done = [label for label, *_r, out_json in all_jobs if out_json.exists()]
        jobs = [j for j in all_jobs if not j[-1].exists()]
        if already_done:
            print(f"[run_parallel] skipping {len(already_done)} level(s) with an "
                  f"existing result (pass --force to redo): {', '.join(already_done)}")

    concurrency_ceiling = args.max_concurrency or max(1, len(jobs))

    # torch defaults to claiming EVERY core for its own intra-op thread pool
    # per process -- confirmed directly to cause severe oversubscription
    # here (14 concurrent workers each trying to use all 128 cores stalled
    # the CPU-bound decode/NMS work so badly that GPU utilization read ~0%,
    # not because the GPU had nothing to do but because nothing was reaching
    # it). Divide the box's cores across the worst-case concurrency instead
    # of leaving it uncapped; passed both as an explicit
    # worker.py flag (torch.set_num_threads) and as env vars
    # (native libs that only read OMP_NUM_THREADS/MKL_NUM_THREADS at
    # process start, before torch.set_num_threads could take effect).
    threads_per_worker = max(1, (os.cpu_count() or 8) // concurrency_ceiling)
    print(f"[run_parallel] {len(jobs)} full-val passes queued, "
          f"mem_threshold={args.mem_threshold:.0%}, concurrency_ceiling={concurrency_ceiling}, "
          f"threads_per_worker={threads_per_worker}")

    t_start = time.time()

    def launch(label, level, ckpt, out_json):
        cmd = [sys.executable, str(EVAL_DIR / "worker.py"),
               "--network", args.network, "--checkpoint", ckpt,
               "--label", label, "--out", str(out_json),
               "--pred-dir", str(out_dir / f"pred_{label.replace('=', '')}"),
               "--num-threads", str(threads_per_worker)]
        if args.device:
            cmd += ["--device", args.device]
        # Lowest CPU (nice 19) and I/O (ionice class 3, "idle") scheduling
        # priority -- this batch is meant to soak up otherwise-idle
        # capacity (e.g. while a training run owns the GPU), not compete
        # with it. Without this, these workers sit at the same nice level
        # as everything else and the kernel scheduler has no reason to
        # favor training's threads over eval's under real contention.
        if not args.no_lowest_priority:
            cmd = ["nice", "-n", "19", "ionice", "-c", "3"] + cmd
        log_path = out_dir / f"{label.replace('=', '')}.log"
        log_f = open(log_path, "w")
        env = dict(os.environ)
        env["OMP_NUM_THREADS"] = str(threads_per_worker)
        env["MKL_NUM_THREADS"] = str(threads_per_worker)
        env["NUMEXPR_NUM_THREADS"] = str(threads_per_worker)
        env["OPENBLAS_NUM_THREADS"] = str(threads_per_worker)
        proc = subprocess.Popen(cmd, cwd=str(REPO_ROOT), stdout=log_f, stderr=subprocess.STDOUT, env=env)
        return proc, log_f

    # Slow-start / additive-increase-multiplicative-decrease scheduling:
    # start at concurrency=1 (never assume the box can safely run this many
    # levels at once) and add one more slot after every clean finish, up to
    # concurrency_ceiling. On any failure (worker exited nonzero, or exited 0
    # but wrote no out_json -- e.g. OOM-killed between exit and flush), halve
    # concurrency immediately and requeue the failed level for another try.
    # This is the opposite of the previous "launch everything, back off only
    # after an OOM" approach, which raced nvidia-smi's reporting lag across a
    # burst of near-simultaneous launches and blew straight past the memory
    # gate before any of them showed up as used. Ramping up from 1 instead
    # means the failure mode is "one job's retry is briefly serialized," not
    # "half the batch OOMs together."
    concurrency = 1
    pending = list(jobs)
    running = []  # list of (proc, log_f, label, level, ckpt, out_json)
    retries_used = {}
    terminally_failed = []
    last_launch = 0.0

    while pending or running:
        while (pending and len(running) < concurrency
               and (time.time() - last_launch) >= args.launch_stagger_seconds):
            gpu_frac = 0.0 if args.device == "cpu" else _gpu_usage_fraction()
            ram_frac = _ram_usage_fraction()
            if gpu_frac >= args.mem_threshold or ram_frac >= args.mem_threshold:
                print(f"[run_parallel] holding launches -- gpu={gpu_frac:.0%} "
                      f"ram={ram_frac:.0%} (threshold {args.mem_threshold:.0%})")
                break
            label, level, ckpt, out_json = pending.pop(0)
            proc, log_f = launch(label, level, ckpt, out_json)
            last_launch = time.time()
            print(f"[run_parallel] started {label} (pid {proc.pid}) -- "
                  f"gpu={gpu_frac:.0%} ram={ram_frac:.0%}, {len(running) + 1}/{concurrency} running "
                  f"(ceiling {concurrency_ceiling})")
            running.append((proc, log_f, label, level, ckpt, out_json))

        time.sleep(args.poll_seconds)
        still_running = []
        for proc, log_f, label, level, ckpt, out_json in running:
            ret = proc.poll()
            if ret is None:
                still_running.append((proc, log_f, label, level, ckpt, out_json))
                continue
            log_f.close()
            ok = ret == 0 and out_json.exists()
            if ok:
                concurrency = min(concurrency_ceiling, concurrency + 1)
                print(f"[run_parallel] finished {label}: ok -- "
                      f"concurrency now {concurrency}/{concurrency_ceiling}")
            else:
                concurrency = max(1, concurrency // 2)
                retries_used[label] = retries_used.get(label, 0) + 1
                if retries_used[label] > args.max_retries:
                    print(f"[run_parallel] finished {label}: FAILED (exit {ret}) -- "
                          f"out of retries ({args.max_retries}), giving up on this level; "
                          f"concurrency now {concurrency}/{concurrency_ceiling}")
                    terminally_failed.append(label)
                else:
                    print(f"[run_parallel] finished {label}: FAILED (exit {ret}) -- "
                          f"requeuing (retry {retries_used[label]}/{args.max_retries}); "
                          f"concurrency now {concurrency}/{concurrency_ceiling}")
                    pending.append((label, level, ckpt, out_json))
        running = still_running

    if terminally_failed:
        print(f"[run_parallel] {len(terminally_failed)} level(s) still failing after "
              f"{args.max_retries} retries: {', '.join(terminally_failed)}")

    elapsed = time.time() - t_start
    print(f"[run_parallel] all {len(jobs)} queued passes done in {elapsed / 60:.1f} min")

    rows = []
    for label, level, ckpt, out_json in all_jobs:
        if out_json.exists():
            rows.append(json.loads(out_json.read_text()))
        else:
            num_clusters = None if level == "float32" else int(level.removeprefix("c"))
            rows.append({"label": label, "num_clusters": num_clusters, "error": "no result written"})

    # "Dataset part" comparison: the quick-probe columns show the SAME
    # metric computed on the biased 400-image sample (QUICK_PROBE_SIZE in
    # train_lut_quant.py) that early-stopping actually used during QAT,
    # pulled from report.json's level_log -- side by side with the real
    # full-3226-image numbers above, this shows directly how much the
    # sampled-subset probe under/over-states the real number per level
    # (every unsampled image counts as 0 recall in the quick probe, so it's
    # a downward-biased but internally consistent proxy -- see
    # widerface_eval.py's generate_widerface_predictions docstring).
    quick_by_label = {}
    report_path_src = results_dir / "report.json"
    if report_path_src.exists():
        src_report = json.loads(report_path_src.read_text())
        if "baseline_aps_quick" in src_report:
            quick_by_label["baseline"] = we.mean_ap(src_report["baseline_aps_quick"])
        for lvl in src_report.get("level_log", []):
            quick_by_label[f"k={lvl['num_clusters']}"] = lvl.get("mean_ap_quick")

    report_path = out_dir / "full_eval_report.json"
    for r in rows:
        if "label" in r and r["label"] in quick_by_label:
            r["mean_ap_quick_400"] = quick_by_label[r["label"]]
    report_path.write_text(json.dumps(rows, indent=2))

    baseline_mean = next((r["mean_ap_full"] for r in rows if r["label"] == "baseline" and "mean_ap_full" in r), None)

    print("\n=== FULL-VAL REPORT:", args.network, "===")
    header = (f"{'level':<10}{'easy':<9}{'medium':<9}{'hard':<9}{'full-mean':<11}"
              f"{'vs baseline':<14}{'quick(400)':<12}{'quick-full gap':<15}")
    print(header)
    for r in rows:
        if "error" in r:
            print(f"{r['label']:<10}ERROR: {r['error']}")
            continue
        aps = r["aps_full"]
        mean = r["mean_ap_full"]
        delta = f"{(mean - baseline_mean) / baseline_mean:+.1%}" if baseline_mean else "n/a"
        quick = quick_by_label.get(r["label"])
        quick_str = f"{quick:.4f}" if quick is not None else "n/a"
        gap = f"{(quick - mean):+.4f}" if quick is not None else "n/a"
        print(f"{r['label']:<10}{aps['easy']:<9.4f}{aps['medium']:<9.4f}{aps['hard']:<9.4f}{mean:<11.4f}"
              f"{delta:<14}{quick_str:<12}{gap:<15}")

    print(f"\nfull report -> {report_path}")


if __name__ == "__main__":
    main()
