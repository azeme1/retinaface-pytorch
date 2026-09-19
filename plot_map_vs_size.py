"""Plots mean AP vs. estimated size (MB) per backbone/level from the metrics tables in
external/retinaface-xs/README.md (the HF model repo README). Writes map_vs_size_{linear,logx,loglog}_{light,dark}.png and map_vs_size.csv."""
import csv, json
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

HERE = Path(__file__).resolve().parent

def read_readme_metrics(path):
    """Parses the per-backbone tables between the METRICS TABLE markers of the HF model repo README.
    Columns: clusters | easy | medium | hard | mean | drop | est. size (MB) | compression ratio | checkpoint;
    the selected level's row is bold."""
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
                mean=float(c[4]), estSizeMb=float(c[6]), ratio=c[7], selected=sel)
    return out


m = read_readme_metrics(HERE.parent / "retinaface-xs" / "README.md")
order = ["mobilenetv1", "mobilenetv1_0.25", "mobilenetv1_0.50", "mobilenetv2", "resnet18", "resnet34", "resnet50"]
LABEL_OFFSET = {"mobilenetv1": (18, -34)}  # default (-8, 9); this one sits under a dense cluster of other lines
THEMES = {  # validated categorical palette, fixed slot order (dataviz skill / references/palette.md)
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", grid="#e6e5e0", series=["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"]),
    "dark": dict(surface="#1a1a19", ink="#ffffff", ink2="#c3c2b7", grid="#333331", series=["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9"]),
}
with open(HERE / "map_vs_size.csv", "w", newline="") as f:  # table view of every plotted point
    w = csv.writer(f); w.writerow(["backbone", "level", "mean_ap_percent", "est_size_mb", "compression_ratio", "selected"])
    for n in order:
        for lvl, v in m[n].items():
            w.writerow([n, lvl, round(v["mean"] * 100, 2), v["estSizeMb"], v["ratio"], v["selected"]])

SCALES = {  # name -> (x scale, y scale, subtitle fragment)
    "linear": ("linear", "linear", "linear axes"),
    "logx": ("log", "linear", "log x-axis"),
    "loglog": ("log", "log", "log-log axes"),
}
XT_LOG = [0.1, 0.3, 1, 3, 10, 30, 100]
YT_LOG = [55, 60, 65, 70, 75, 80, 85, 90, 95]


def draw_series(ax, t, labels, small=False):
    for i, n in enumerate(order):
        col = t["series"][i]
        pts = sorted((v["estSizeMb"], v["mean"] * 100, k, v["selected"]) for k, v in m[n].items())
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=col, lw=2, solid_capstyle="round", zorder=2)
        cl = [p for p in pts if p[2] != "float32"]; fl = [p for p in pts if p[2] == "float32"]
        ax.scatter([p[0] for p in cl], [p[1] for p in cl], s=26 if small else 38, color=col, edgecolor=t["surface"], linewidth=1.5, zorder=3)
        ax.scatter([p[0] for p in fl], [p[1] for p in fl], s=50 if small else 70, marker="D", color=col, edgecolor=t["surface"], linewidth=1.5, zorder=4)
        for p in pts:
            if p[3]:
                ax.scatter([p[0]], [p[1]], s=110 if small else 150, facecolor="none", edgecolor=col, linewidth=2, zorder=5)
                if labels:
                    dx, dy = LABEL_OFFSET.get(n, (-8, 9))
                    ax.annotate(p[2], (p[0], p[1]), xytext=(dx, dy), textcoords="offset points",
                                ha="right" if dx < 0 else "left", fontsize=9, color=t["ink2"],
                                arrowprops=dict(arrowstyle="-", color=col, lw=1, shrinkA=1, shrinkB=7) if n in LABEL_OFFSET else None)


def style_axes(ax, t, small=False):
    ax.set_facecolor(t["surface"])
    ax.grid(True, color=t["grid"], lw=0.8, zorder=0); ax.set_axisbelow(True)
    for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
    for s_ in ("left", "bottom"): ax.spines[s_].set_color(t["grid"])
    ax.tick_params(colors=t["ink2"], labelsize=8 if small else 10)


for scale, (xs, ys, frag) in SCALES.items():
    for mode, t in THEMES.items():
        fig, ax = plt.subplots(figsize=(11, 6.6), dpi=160)
        fig.patch.set_facecolor(t["surface"]); style_axes(ax, t)
        draw_series(ax, t, labels=(scale != "linear"))
        ax.set_xscale(xs); ax.set_yscale(ys)
        if xs == "log":
            ax.xaxis.set_major_locator(FixedLocator(XT_LOG)); ax.xaxis.set_minor_locator(NullLocator())
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
            ax.set_xlim(0.08, 140)
        else:
            ax.set_xlim(-2, 110)
        if ys == "log":
            ax.yaxis.set_major_locator(FixedLocator(YT_LOG)); ax.yaxis.set_minor_locator(NullLocator())
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.set_ylim(55, 95)
        ax.set_xlabel("Estimated model size (MB" + (", log scale)" if xs == "log" else ")"), color=t["ink2"], fontsize=11)
        ax.set_ylabel("Mean AP on WIDER FACE val (%" + (", log scale)" if ys == "log" else ")"), color=t["ink2"], fontsize=11)
        fig.text(0.075, 0.955, f"Accuracy vs. estimated size, by backbone and compression level ({frag})", color=t["ink"], fontsize=15, fontweight="bold", ha="left")
        sub = "Mean of easy/medium/hard AP on the full val set. One line per backbone; each dot is a cluster level (2-256).\nRinged dot = the selected level, diamond = float32 baseline."
        if scale == "linear":
            sub += " Inset: zoom on sizes up to 6 MB, where the selected levels are."
        fig.text(0.075, 0.918, sub, color=t["ink2"], fontsize=9.5, ha="left", va="top")
        handles = [Line2D([0], [0], color=t["series"][i], lw=2, marker="o", markersize=6, markeredgecolor=t["surface"], label=n) for i, n in enumerate(order)]
        if scale == "linear":
            leg = ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.09, 0.05), frameon=False, fontsize=10, labelcolor=t["ink"], title="Backbone", title_fontsize=10)
            ins = ax.inset_axes([0.47, 0.07, 0.51, 0.52])
            style_axes(ins, t, small=True); ins.set_facecolor(t["surface"])
            draw_series(ins, t, labels=True, small=True)
            ins.set_xlim(0, 6); ins.set_ylim(55, 95)
            ins.patch.set_edgecolor(t["grid"])
            ins.set_title("zoom: 0-6 MB", loc="left", fontsize=9, color=t["ink2"])
            # selected-level labels in the inset: nudge to avoid the cluster near 1-2 MB
        else:
            leg = ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=10, labelcolor=t["ink"], title="Backbone", title_fontsize=10)
        leg.get_title().set_color(t["ink2"])
        fig.subplots_adjust(left=0.075, right=0.975, top=0.85, bottom=0.11)
        fig.savefig(HERE / f"map_vs_size_{scale}_{mode}.png", facecolor=fig.get_facecolor()); plt.close(fig)
print("saved")
