"""Re-picks each backbone's final LUT level using the REAL full WIDER FACE
validation AP (from evaluate/run_parallel.py's full_eval_report.json)
instead of the 400-image quick-probe train_lut_quant.py used for its own
early-stop/pick_best_level selection during training.

Selection rule: among every level whose full-val mean AP is within 1% of the
float32 baseline's full-val mean AP (relative drop, i.e.
mean_ap_full >= baseline_mean_ap_full * (1 - tolerance)), pick the one with
the highest compression ratio (smallest num_clusters, ties broken by
compression_ratio from report.json's level_log). This does NOT overwrite
report.json's existing quick-probe-based "selected_level" -- it adds a
separate "selected_level_full_val" field so both selections stay inspectable
side by side.

Usage:
    python select_by_full_val.py --results-dir results/mobilenetv1
    python select_by_full_val.py --all  # every results/<network> with both
                                         # report.json and a full_eval_report
"""
import argparse
import json
from pathlib import Path

RESULTS_ROOT = Path(__file__).resolve().parent / "results"
NETWORKS = ["mobilenetv1_0.25", "mobilenetv1_0.50", "mobilenetv1", "mobilenetv2",
            "resnet18", "resnet34", "resnet50"]


def select_one(results_dir: Path, tolerance: float = 0.01):
    report_path = results_dir / "report.json"
    full_report_path = results_dir / "full_eval_parallel" / "full_eval_report.json"
    if not report_path.exists() or not full_report_path.exists():
        return None

    report = json.loads(report_path.read_text())
    full_rows = json.loads(full_report_path.read_text())

    compression_by_k = {
        lvl["num_clusters"]: {
            "compression_ratio": lvl.get("compression_ratio"),
            "compressed_mb": lvl.get("compressed_mb"),
        }
        for lvl in report.get("level_log", [])
    }

    baseline_row = next((r for r in full_rows if r["label"] == "baseline" and "mean_ap_full" in r), None)
    if baseline_row is None:
        return {"network": report["network"], "error": "no successful baseline full-val row"}
    baseline_mean = baseline_row["mean_ap_full"]
    threshold = baseline_mean * (1 - tolerance)

    candidates = []
    for r in full_rows:
        if r["label"] == "baseline" or "mean_ap_full" not in r or r.get("num_clusters") is None:
            continue
        k = r["num_clusters"]
        comp = compression_by_k.get(k, {})
        candidates.append({
            "num_clusters": k,
            "mean_ap_full": r["mean_ap_full"],
            "aps_full": r["aps_full"],
            "compression_ratio": comp.get("compression_ratio"),
            "compressed_mb": comp.get("compressed_mb"),
            "meets_tolerance": r["mean_ap_full"] >= threshold,
        })
    candidates.sort(key=lambda c: c["num_clusters"], reverse=True)  # coarsest (safest) first

    passing = [c for c in candidates if c["meets_tolerance"] and c["compression_ratio"] is not None]
    selected = max(passing, key=lambda c: c["compression_ratio"]) if passing else None

    result = {
        "network": report["network"],
        "baseline_mean_ap_full": baseline_mean,
        "tolerance": tolerance,
        "threshold_mean_ap_full": threshold,
        "candidates": candidates,
        "selected_level_full_val": selected,
    }

    report["selected_level_full_val"] = selected
    report["full_val_selection_baseline"] = baseline_mean
    report_path.write_text(json.dumps(report, indent=2))
    return result


def print_table(result):
    net = result["network"]
    if "error" in result:
        print(f"\n=== {net}: {result['error']} ===")
        return
    print(f"\n=== {net}: full-val-based selection (tolerance {result['tolerance']:.0%}, "
          f"baseline mean AP {result['baseline_mean_ap_full']:.4f}, "
          f"threshold {result['threshold_mean_ap_full']:.4f}) ===")
    header = f"{'k':<6}{'mean_ap_full':<14}{'easy':<9}{'medium':<9}{'hard':<9}{'compression':<13}{'ok?':<6}"
    print(header)
    for c in result["candidates"]:
        comp = f"{c['compression_ratio']:.2f}x" if c["compression_ratio"] else "n/a"
        ok = "PASS" if c["meets_tolerance"] else "drop"
        aps = c["aps_full"]
        print(f"{c['num_clusters']:<6}{c['mean_ap_full']:<14.4f}{aps['easy']:<9.4f}"
              f"{aps['medium']:<9.4f}{aps['hard']:<9.4f}{comp:<13}{ok:<6}")
    sel = result["selected_level_full_val"]
    if sel:
        print(f"--> selected: k={sel['num_clusters']} "
              f"(mean AP {sel['mean_ap_full']:.4f}, {sel['compression_ratio']:.2f}x compression)")
    else:
        print("--> no level meets the tolerance -- no selection made")


def main():
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--results-dir")
    g.add_argument("--all", action="store_true")
    p.add_argument("--tolerance", type=float, default=0.01,
                    help="allowed relative drop vs baseline full-val mean AP (default 0.01 = 1%%)")
    args = p.parse_args()

    dirs = [Path(args.results_dir)] if args.results_dir else [RESULTS_ROOT / n for n in NETWORKS]
    for d in dirs:
        result = select_one(d, tolerance=args.tolerance)
        if result is None:
            print(f"\n=== {d.name}: no report.json / full_eval_report.json yet, skipping ===")
            continue
        print_table(result)


if __name__ == "__main__":
    main()
