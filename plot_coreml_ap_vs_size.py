"""AP vs. MEASURED CoreML size, from the CoreML artifacts themselves (no estimates, no averaging).
AP  = the CoreML model's own easy/medium/hard AP from the macOS validation (checkpoint-coreml.md; fixed 640x640 letterbox, full val set).
Size = the published CoreML file on HF: zip (what you download) or raw (unpacked model.mlpackage), read from the HF repo.
Writes coreml_ap_vs_size_{zip,raw}_{logx,loglog}_{light,dark}.png and coreml_ap_vs_size.csv."""
import csv, re, zipfile
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
from huggingface_hub import HfApi, HfFileSystem

HERE = Path(__file__).resolve().parent
REPO = "azemel/retinaface-xs"
order = ["mobilenetv1", "mobilenetv1_0.25", "mobilenetv1_0.50", "mobilenetv2", "resnet18", "resnet34", "resnet50"]
SUBSETS = [("easy", "Easy (7,211 faces)"), ("medium", "Medium (13,319 faces)"), ("hard", "Hard (31,958 faces)")]
THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", grid="#e6e5e0", series=["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"]),
    "dark": dict(surface="#1a1a19", ink="#ffffff", ink2="#c3c2b7", grid="#333331", series=["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9"]),
}

# ---- AP of the CoreML models (macOS validation)
ap = {}
for line in (HERE / "checkpoint-coreml.md").read_text().splitlines():
    c = [x.strip() for x in line.strip().strip("|").split("|")]
    if line.startswith("| ") and len(c) == 8 and re.match(r"^[\d.]+ / [\d.]+$", c[2]):
        ap[(c[0], c[1])] = dict(zip(("easy", "medium", "hard"), (float(x.split(" / ")[1]) for x in c[2:5])))
# ---- measured sizes (zip on HF, and unpacked package read from the zip directory)
info = {s.rfilename: s.size for s in HfApi().repo_info(REPO, files_metadata=True).siblings}
fs = HfFileSystem(); size = {}
for k in ap:
    lv = k[1] if k[1] == "float32" else "c" + k[1]
    path = f"checkpoints/{k[0]}/coreml/{k[0]}_{lv}.zip"
    with fs.open(f"{REPO}/{path}", "rb") as f:
        raw = sum(i.file_size for i in zipfile.ZipFile(f).infolist())
    size[k] = dict(zip=info[path] / 1e6, raw=raw / 1e6)
# selected level per backbone (from the README's bold rows)
sel = set()
for l in (HERE.parent / "retinaface-xs" / "README.md").read_text().splitlines():
    c = [x.strip() for x in l.strip().strip("|").split("|")]
    if l.startswith("| ") and len(c) == 4 and c[0].replace("*", "") in order and "MB" in c[2] and "**" in c[1]:
        sel.add((c[0].replace("*", ""), c[1].replace("*", "")))
key = lambda k: (order.index(k[0]), k[1] == "float32", 0 if k[1] == "float32" else int(k[1]))
with open(HERE / "coreml_ap_vs_size.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["backbone", "level", "easy_ap_percent", "medium_ap_percent", "hard_ap_percent", "coreml_zip_mb", "coreml_raw_mb", "selected"])
    for k in sorted(ap, key=key):
        w.writerow([k[0], k[1], *(ap[k][s] for s in ("easy", "medium", "hard")), round(size[k]["zip"], 3), round(size[k]["raw"], 3), k in sel])

ALL = [v for d in ap.values() for v in d.values()]
LO, HI = min(ALL), max(ALL)
XT = [0.1, 0.3, 1, 3, 10, 30, 100]
VARIANTS = [("zip", "logx"), ("zip", "loglog"), ("raw", "loglog")]
XLAB = {"zip": "CoreML package size, zipped (MB, log scale)", "raw": "CoreML package size, unpacked (MB, log scale)"}
for which, scale in VARIANTS:
    ylog = scale == "loglog"
    xs = [size[k][which] for k in ap]; XLO, XHI = min(xs) * 0.8, max(xs) * 1.3
    for mode, t in THEMES.items():
        fig, axes = plt.subplots(1, 3, figsize=(17, 6), dpi=140, sharex=True, sharey=True)
        fig.patch.set_facecolor(t["surface"])
        for ax, (sub, title) in zip(axes, SUBSETS):
            ax.set_facecolor(t["surface"])
            for i, n in enumerate(order):
                col = t["series"][i]
                pts = sorted(((size[k][which], ap[k][sub], k[1], k in sel) for k in ap if k[0] == n))
                ax.plot([p[0] for p in pts], [p[1] for p in pts], color=col, lw=2, solid_capstyle="round", zorder=2)
                cl = [p for p in pts if p[2] != "float32"]; fl = [p for p in pts if p[2] == "float32"]
                ax.scatter([p[0] for p in cl], [p[1] for p in cl], s=30, color=col, edgecolor=t["surface"], linewidth=1.2, zorder=3)
                ax.scatter([p[0] for p in fl], [p[1] for p in fl], s=55, marker="D", color=col, edgecolor=t["surface"], linewidth=1.2, zorder=4)
                for p in pts:
                    if p[3]: ax.scatter([p[0]], [p[1]], s=115, facecolor="none", edgecolor=col, linewidth=2, zorder=5)
            ax.set_xscale("log"); ax.set_xlim(XLO, XHI)
            ax.xaxis.set_major_locator(FixedLocator([x for x in XT if XLO <= x <= XHI])); ax.xaxis.set_minor_locator(NullLocator())
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
            if ylog:
                ax.set_yscale("log"); ax.set_ylim(LO * 0.95, HI * 1.03)
                ax.yaxis.set_major_locator(FixedLocator([x for x in (20, 25, 30, 40, 50, 60, 70, 80, 90) if LO * 0.95 <= x <= HI * 1.03]))
                ax.yaxis.set_minor_locator(NullLocator()); ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
            else:
                ax.set_ylim(LO - 3, HI + 3)
            ax.grid(True, color=t["grid"], lw=0.8, zorder=0); ax.set_axisbelow(True)
            for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
            for s_ in ("left", "bottom"): ax.spines[s_].set_color(t["grid"])
            ax.tick_params(colors=t["ink2"], labelsize=9)
            ax.set_title(title, loc="left", color=t["ink"], fontsize=12, fontweight="bold")
            ax.set_xlabel(XLAB[which], color=t["ink2"], fontsize=10)
            if ax is axes[0]: ax.set_ylabel("CoreML AP (%" + (", log scale)" if ylog else ")"), color=t["ink2"], fontsize=10)
        handles = [Line2D([0], [0], color=t["series"][i], lw=2, marker="o", markersize=6, markeredgecolor=t["surface"], label=n) for i, n in enumerate(order)]
        fig.legend(handles=handles, loc="lower center", ncol=7, frameon=False, fontsize=10, labelcolor=t["ink"], bbox_to_anchor=(0.5, 0.0))
        fig.text(0.04, 0.965, f"CoreML models: AP vs. measured package size, per subset ({'log-log' if ylog else 'log x-axis'})", color=t["ink"], fontsize=15, fontweight="bold", ha="left")
        fig.text(0.04, 0.925, "AP of the CoreML artifacts (macOS validation, fixed 640x640 letterbox, full val set); size measured on the published files. Easy / medium / hard are never averaged. Ringed = selected level, diamond = float32. Shared axes.", color=t["ink2"], fontsize=9, ha="left")
        fig.subplots_adjust(left=0.05, right=0.985, top=0.86, bottom=0.2, wspace=0.22)
        fig.savefig(HERE / f"coreml_ap_vs_size_{which}_{scale}_{mode}.png", facecolor=fig.get_facecolor()); plt.close(fig)
print("saved", len(ap), "points")
