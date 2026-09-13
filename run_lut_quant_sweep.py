"""Orchestrates train_lut_quant.py across all 7 RetinaFace backbones,
one subprocess per backbone (crash/OOM/cudnn-cache isolation between runs,
same rationale as the main llwll repo's scratches_detector/utils.py
SweepRunner), sequential since this box has a single GPU. Skips a backbone
whose report.json already exists (resumable across interrupted sessions).
Writes a combined summary.json + prints a results table at the end.

Results land in results/ alongside this script (gitignored -- see
.gitignore's "results/" entry), same as this repo's own train.py convention.
"""

import json
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_ROOT = SCRIPT_DIR / "results"

# (network, pretrained weights path relative to external/retinaface-pytorch, or None,
#  batch-size override or None to use config.py's per-backbone default (32 for every
#  backbone here except resnet50's own 8)).
#
# mobilenetv1 (full width) and mobilenetv2 both OOM'd at batch_size=32 immediately
# after mobilenetv1_0.50 finished (confirmed via traceback: torch.OutOfMemoryError
# in BatchNorm2d.forward, ~15.3-15.4GB in use on this 16GB card) -- unlike the
# 0.25/0.50-width mobilenetv1 variants, which trained fine at the same batch size,
# these are full-width backbones with meaningfully larger per-batch activations.
# Halved to 16 for both -- mobilenetv1 then ran its entire 13-level schedule clean
# at 16, but mobilenetv2 OOM'd again anyway (traceback: torch.OutOfMemoryError in
# a residual-block forward, ~15.26GB in use, mid-training not just eval), meaning
# its inverted-residual expansion blocks have a meaningfully heavier per-batch
# footprint than mobilenetv1's plain depthwise-separable ones even at the same
# batch size -- halved again to 8. resnet34 hasn't run yet but is the next step up
# from resnet18 (which used ~10.5-12GB at batch_size=32) with no OOM margin to
# spare if it runs even somewhat heavier, so it's knocked down preemptively too
# rather than waiting to find out via another crash; resnet50 was already reduced
# to 8 from the start (the actual biggest backbone here).
BACKBONES = [
    ("mobilenetv1_0.25", "weights/retinaface_mv1_0.25.pth", None),
    ("mobilenetv1_0.50", "weights/retinaface_mv1_0.50.pth", None),
    ("mobilenetv1", "weights/retinaface_mv1.pth", 16),
    ("mobilenetv2", "weights/retinaface_mv2.pth", 8),
    ("resnet18", "weights/retinaface_r18.pth", None),
    ("resnet34", "weights/retinaface_r34.pth", 16),
    # batch_size stays reduced even though this now skips from-scratch
    # baseline training (see weights/retinaface_r50_biubug6.pth's own
    # docstring/history) -- the per-step memory footprint during the LUT
    # annealing/fine-tuning stage depends on the model's own size, not on
    # how the float32 baseline was obtained, and resnet50 is still the
    # biggest backbone in this sweep.
    ("resnet50", "weights/retinaface_r50_biubug6.pth", 8),
]


def main():
    RESULTS_ROOT.mkdir(exist_ok=True)
    summary = {}

    for network, pretrained, batch_size in BACKBONES:
        results_dir = RESULTS_ROOT / network
        report_path = results_dir / "report.json"
        if report_path.exists():
            print(f"=== {network}: report.json already exists, skipping ===")
            summary[network] = json.loads(report_path.read_text())
            continue

        # --schedule-stop-drop 1e9 disables the per-backbone "stop the whole
        # cluster-count walk once quality drops too far" early exit, so
        # EVERY backbone gets tested at every level of the shared schedule
        # (256,128,64,32,16,12,8,7,6,5,4,3,2) -- lets results be compared
        # across backbones at the same cluster counts, not just each
        # backbone's own best/first-good level. Costs more time per
        # backbone (back to walking the full schedule, no early cutoff).
        cmd = [sys.executable, str(SCRIPT_DIR / "train_lut_quant.py"), "--network", network,
               "--results-dir", str(results_dir), "--schedule-stop-drop", "1e9"]
        if pretrained:
            cmd += ["--pretrained", pretrained]
        if batch_size is not None:
            cmd += ["--batch-size", str(batch_size)]

        print(f"=== {network}: starting ({'pretrained' if pretrained else 'from-scratch baseline'}) ===")
        t0 = time.time()
        log_path = results_dir / "run.log"
        results_dir.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w") as log_f:
            proc = subprocess.run(cmd, cwd=str(SCRIPT_DIR), stdout=log_f, stderr=subprocess.STDOUT)
        elapsed = time.time() - t0
        print(f"=== {network}: exited with code {proc.returncode} in {elapsed / 60:.1f} min "
              f"(see {log_path}) ===")

        if report_path.exists():
            summary[network] = json.loads(report_path.read_text())
        else:
            summary[network] = {"error": f"no report.json produced, exit code {proc.returncode}"}

    with open(RESULTS_ROOT / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== SWEEP SUMMARY ===")
    header = (f"{'network':<18}{'fp32 mAP(E/M/H)':<24}{'quant mAP(E/M/H)':<24}"
              f"{'selected k':<12}{'ratio':<8}{'total epochs':<14}")
    print(header)
    for network, rep in summary.items():
        if "error" in rep:
            print(f"{network:<18}ERROR: {rep['error']}")
            continue
        b = rep.get("baseline_aps_full", {})
        q = rep.get("final_aps_full", {})
        ratio = rep.get("compression", {}).get("compression_ratio", 0.0)
        selected_k = rep.get("selected_level", {}).get("num_clusters", "?")
        total_epochs = sum(lvl.get("epochs_run", 0) for lvl in rep.get("level_log", []))
        b_str = f"{b.get('easy', 0):.3f}/{b.get('medium', 0):.3f}/{b.get('hard', 0):.3f}"
        q_str = f"{q.get('easy', 0):.3f}/{q.get('medium', 0):.3f}/{q.get('hard', 0):.3f}"
        print(f"{network:<18}{b_str:<24}{q_str:<24}{selected_k:<12}{ratio:<8.2f}{total_epochs:<14}")


if __name__ == "__main__":
    main()
