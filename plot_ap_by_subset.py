"""Plots the WIDER FACE Easy / Medium / Hard AP vs. estimated size (MB) as three separate panels (no averaging
across subsets -- they score different numbers of faces), from the metrics tables in external/retinaface-xs/README.md.
Writes ap_by_subset_{logx,loglog}_{light,dark}.png and ap_by_subset.csv."""
import csv
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

HERE = Path(__file__).resolve().parent
order = ["mobilenetv1", "mobilenetv1_0.25", "mobilenetv1_0.50", "mobilenetv2", "resnet18", "resnet34", "resnet50"]
SUBSETS = [("easy", "Easy (7,211 faces)"), ("medium", "Medium (13,319 faces)"), ("hard", "Hard (31,958 faces)")]
THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", grid="#e6e5e0", series=["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"]),
    "dark": dict(surface="#1a1a19", ink="#ffffff", ink2="#c3c2b7", grid="#333331", series=["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9"]),
}


def read_readme_metrics(path):
    text = Path(path).read_text()
    block = text[text.index("<!-- METRICS TABLE START -->"):text.index("<!-- METRICS TABLE END -->")]
    out, net = {}, None
    for line in block.splitlines():
        if line.startswith("### "):
            net = line[4:].strip(); out[net] = {}
        elif line.startswith("| ") and net and not line.startswith("| clusters") and "---" not in line:
            c = [x.strip() for x in line.strip().strip("|").split("|")]
            sel = c[0].startswith("**"); c = [x.strip("*") for x in c]
            out[net]["float32" if c[0] == "float32" else "c" + c[0]] = dict(
                easy=float(c[1]) * 100, medium=float(c[2]) * 100, hard=float(c[3]) * 100, estSizeMb=float(c[6]), selected=sel)
    return out


m = read_readme_metrics(HERE.parent / "retinaface-xs" / "README.md")
with open(HERE / "ap_by_subset.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["backbone", "level", "easy_ap_percent", "medium_ap_percent", "hard_ap_percent", "est_size_mb", "selected"])
    for n in order:
        for lvl, v in m[n].items():
            w.writerow([n, lvl, round(v["easy"], 2), round(v["medium"], 2), round(v["hard"], 2), v["estSizeMb"], v["selected"]])

ALLVALS = [v[k] for n in order for v in m[n].values() for k in ("easy", "medium", "hard")]
LO, HI = min(ALLVALS), max(ALLVALS)  # one y-range shared by all three panels
SCALES = {"logx": ("log", "linear", "log x-axis"), "loglog": ("log", "log", "log-log axes")}
for scale, (xs, ys, frag) in SCALES.items():
  for mode, t in THEMES.items():
    fig, axes = plt.subplots(1, 3, figsize=(17, 6), dpi=140, sharex=True, sharey=True)
    fig.patch.set_facecolor(t["surface"])
    for ax, (key, title) in zip(axes, SUBSETS):
        ax.set_facecolor(t["surface"])
        for i, n in enumerate(order):
            col = t["series"][i]
            pts = sorted((v["estSizeMb"], v[key], k, v["selected"]) for k, v in m[n].items())
            ax.plot([p[0] for p in pts], [p[1] for p in pts], color=col, lw=2, solid_capstyle="round", zorder=2)
            cl = [p for p in pts if p[2] != "float32"]; fl = [p for p in pts if p[2] == "float32"]
            ax.scatter([p[0] for p in cl], [p[1] for p in cl], s=26, color=col, edgecolor=t["surface"], linewidth=1.2, zorder=3)
            ax.scatter([p[0] for p in fl], [p[1] for p in fl], s=50, marker="D", color=col, edgecolor=t["surface"], linewidth=1.2, zorder=4)
            for p in pts:
                if p[3]: ax.scatter([p[0]], [p[1]], s=110, facecolor="none", edgecolor=col, linewidth=2, zorder=5)
        ax.set_xscale(xs); ax.set_yscale(ys)
        ax.xaxis.set_major_locator(FixedLocator([0.1, 0.3, 1, 3, 10, 30, 100])); ax.xaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.set_xlim(0.08, 140)
        if ys == "log":
            ticks = [x for x in (45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95) if LO - 3 <= x <= HI + 2]
            ax.set_ylim(LO * 0.97, HI * 1.02)
            ax.yaxis.set_major_locator(FixedLocator(ticks)); ax.yaxis.set_minor_locator(NullLocator())
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        else:
            ax.set_ylim(LO - 2, HI + 2)
        ax.grid(True, color=t["grid"], lw=0.8, zorder=0); ax.set_axisbelow(True)
        for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
        for s_ in ("left", "bottom"): ax.spines[s_].set_color(t["grid"])
        ax.tick_params(colors=t["ink2"], labelsize=9)
        ax.set_title(title, loc="left", color=t["ink"], fontsize=12, fontweight="bold")
        ax.set_xlabel("Estimated model size (MB, log scale)", color=t["ink2"], fontsize=10)
        if ax is axes[0]:
            ax.set_ylabel("AP (%" + (", log scale)" if ys == "log" else ")"), color=t["ink2"], fontsize=10)
    handles = [Line2D([0], [0], color=t["series"][i], lw=2, marker="o", markersize=6, markeredgecolor=t["surface"], label=n) for i, n in enumerate(order)]
    fig.legend(handles=handles, loc="lower center", ncol=7, frameon=False, fontsize=10, labelcolor=t["ink"], bbox_to_anchor=(0.5, 0.0))
    fig.text(0.04, 0.965, f"WIDER FACE AP vs. estimated size, per subset (not averaged; {frag})", color=t["ink"], fontsize=15, fontweight="bold", ha="left")
    fig.text(0.04, 0.925, "One line per backbone; each dot is a compression level (2-256 clusters); ringed dot = selected level, diamond = float32. All three panels share the same x and y axes.", color=t["ink2"], fontsize=9.5, ha="left")
    fig.subplots_adjust(left=0.05, right=0.985, top=0.86, bottom=0.2, wspace=0.22)
    fig.savefig(HERE / f"ap_by_subset_{scale}_{mode}.png", facecolor=fig.get_facecolor()); plt.close(fig)
print("saved")
