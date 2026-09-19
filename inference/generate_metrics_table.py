"""Generates the Markdown metrics table used in external/retinaface-xs/README.md
for one RetinaFace backbone's compression sweep: real WIDER FACE full-val
AP (easy / medium / hard, reported separately, never averaged) per cluster
count, the drop of each vs. the float32 baseline, estimated PyTorch weight-only
model size, and compression ratio -- pulled from results/<network>/report.json (per-level compression
estimate) and results/<network>/full_eval_parallel/full_eval_report.json
(real full-val AP per level, produced by this project's private training
pipeline).

Requires both files to already exist for the network (i.e. training AND the
full-val sweep have both completed).

Usage:
    python inference/generate_metrics_table.py --network mobilenetv1
    python inference/generate_metrics_table.py --network mobilenetv1 mobilenetv1_0.25 \\
        --out external/retinaface-xs/README.md --section-marker "<!-- METRICS -->"
"""
import argparse
import json
import os
import re
from pathlib import Path

RESULTS_ROOT = Path(__file__).resolve().parents[1] / "results"
# The raw-checkpoint staging repo (external/retinaface-xs) is a sibling of
# this repo under LLWLL_ROOT, not a subdirectory of it -- its results/<network>/
# pytorch/<network>_cK.zip files are what the "checkpoint" column below links
# to, when they've actually been staged there.
STAGE_ROOT = Path(os.environ["LLWLL_ROOT"]) / "external" / "retinaface-xs"
HF_REPO_ID = "azemel/retinaface-xs"


_HF_FILES = None


def _hf_files() -> set:
    global _HF_FILES
    if _HF_FILES is None:
        from huggingface_hub import HfApi
        _HF_FILES = set(HfApi().list_repo_files(HF_REPO_ID))
    return _HF_FILES


def checkpoint_link(network: str, label: str) -> str:
    """label is "float32" or a cluster-count string like "7". Returns a
    Markdown link (short filename text, pointing at the real HF download
    URL) if that checkpoint has actually been staged locally, else "n/a" --
    never fabricate a link for a file that doesn't exist yet. Staged-locally
    is used as the existence gate (not a live HF check) since this table is
    generated once per sweep and by the time it's staged the corresponding
    push has consistently followed close behind in this project's workflow."""
    fname = f"{network}_{'float32' if label == 'float32' else 'c' + label}.zip"
    rel = f"checkpoints/{network}/pytorch/{fname}"
    if rel not in _hf_files():  # existence gate: is it actually on the HF repo (local staging copies are deleted once identical)
        return "n/a"
    url = f"https://huggingface.co/{HF_REPO_ID}/resolve/main/{rel}"
    return f"[{fname}]({url})"


def build_table(network: str) -> str:
    report = json.loads((RESULTS_ROOT / network / "report.json").read_text())
    full = json.loads((RESULTS_ROOT / network / "full_eval_parallel" / "full_eval_report.json").read_text())

    float32_mb = report["compression"]["float32_mb"]
    comp_by_k = {lvl["num_clusters"]: lvl.get("compressed_mb") for lvl in report["level_log"]}
    selected_k = report["selected_level"]["num_clusters"]

    baseline_row = next(x for x in full if x["label"] == "baseline" and "aps_full" in x)
    base_aps = baseline_row["aps_full"]
    schedule = report["cluster_schedule"]

    lines = [
        f"### {network}",
        "",
        "| clusters | easy | medium | hard | drop easy vs float32 | drop medium vs float32 | drop hard vs float32 | est. size (MB) | compression ratio | checkpoint |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]

    def fmt_row(label, aps, drops, size, ratio, checkpoint, bold=False):
        # easy / medium / hard are three separate metrics (they score different numbers of faces) -- never averaged
        cells = [label, f"{aps['easy']:.4f}", f"{aps['medium']:.4f}", f"{aps['hard']:.4f}", *drops,
                 f"{size:.2f}" if size is not None else "n/a",
                 f"{ratio:.2f}x" if ratio is not None else "n/a"]
        if bold:
            cells = [f"**{c}**" for c in cells]
        cells.append(checkpoint)
        return "| " + " | ".join(cells) + " |"

    lines.append(fmt_row("float32", base_aps, ["—", "—", "—"], float32_mb, 1.0,
                          checkpoint_link(network, "float32")))

    for k in schedule:
        row = next((x for x in full if x["label"] == f"k={k}" and "aps_full" in x), None)
        if row is None:
            continue  # this level has no real full-val result (still pending or failed)
        size = comp_by_k.get(k)
        ratio = float32_mb / size if size else None
        drops = [f"{(row['aps_full'][s_] - base_aps[s_]) / base_aps[s_]:+.2%}" for s_ in ("easy", "medium", "hard")]
        lines.append(fmt_row(str(k), row["aps_full"], drops, size, ratio,
                              checkpoint_link(network, str(k)), bold=(k == selected_k)))

    lines.append("")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--network", nargs="+", required=True)
    p.add_argument("--out", default=None, help="if given, insert/replace the table(s) into this file "
                                                "between --section-marker START/END comments instead of printing")
    p.add_argument("--section-marker", default="METRICS TABLE",
                    help="the block between '<!-- <marker> START -->' and '<!-- <marker> END -->' "
                         "gets replaced in --out")
    args = p.parse_args()

    tables = "\n".join(build_table(net) for net in args.network)

    if not args.out:
        print(tables)
        return

    out_path = Path(args.out)
    text = out_path.read_text()
    start_tag = f"<!-- {args.section_marker} START -->"
    end_tag = f"<!-- {args.section_marker} END -->"
    block = f"{start_tag}\n{tables}\n{end_tag}"

    pattern = re.compile(re.escape(start_tag) + r".*?" + re.escape(end_tag), re.DOTALL)
    if pattern.search(text):
        text = pattern.sub(block, text)
    else:
        text = text.rstrip() + "\n\n" + block + "\n"
    out_path.write_text(text)
    print(f"wrote metrics table(s) for {args.network} -> {out_path}")


if __name__ == "__main__":
    main()
