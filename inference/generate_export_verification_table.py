"""Builds the Markdown "exported formats vs PyTorch" verification table for
external/retinaface-xs/README.md from the per-run logs that
validate_onnx.sh / validate_tflite.sh / validate_tfjs.sh write
(onnx_verify_logs/, tflite_verify_logs/, tfjs_verify_logs/, run from this
directory). Each log holds one export_check.py run: real full-val WIDER FACE
AP of the PyTorch fixed-size wrapper vs. the converted artifact downloaded
from the HF repo.

Usage:
    python generate_export_verification_table.py
    python generate_export_verification_table.py --out ../../retinaface-xs/README.md \\
        --section-marker "EXPORT VERIFICATION"
"""
import argparse
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
FORMATS = [("ONNX", "onnx_verify_logs", "_CUDA"), ("TFLite", "tflite_verify_logs", ""), ("TFJS", "tfjs_verify_logs", "")]
ORDER = ["mobilenetv1", "mobilenetv1_0.25", "mobilenetv1_0.50", "mobilenetv2", "resnet18", "resnet34", "resnet50"]
ROW = re.compile(r"^(Easy|Medium|Hard|Average)\s+([\d.]+)%\s+([\d.]+)%\s+([+-][\d.]+)%", re.M)


def parse(path: Path):
    text = path.read_text().replace("\r", "\n")
    rows = {k: (float(a), float(b)) for k, a, b, _ in ROW.findall(text)}
    return rows if len(rows) == 4 else None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default=None)
    p.add_argument("--section-marker", default="EXPORT VERIFICATION")
    args = p.parse_args()

    data = {}  # (network, level) -> {"PyTorch": mean, fmt: (mean, {E,M,H})}
    for fmt, d, suffix in FORMATS:
        for log in sorted((HERE / d).glob("*.log")):
            name = log.stem[: -len(suffix)] if suffix and log.stem.endswith(suffix) else log.stem
            network, level = name.rsplit("_", 1)
            rows = parse(log)
            if rows is None:
                continue
            e = data.setdefault((network, level), {})
            e["PyTorch"] = rows["Average"][0]
            e[fmt] = rows["Average"][1]

    def keyf(k):
        n, lv = k
        return (ORDER.index(n) if n in ORDER else 99, lv == "float32", 0 if lv == "float32" else int(lv[1:]))

    lines = ["| backbone | clusters | PyTorch mean AP | ONNX (Δ) | TFLite (Δ) | TFJS (Δ) |", "|---|---|---|---|---|---|"]
    worst = {f: 0.0 for f, _, _ in FORMATS}
    for k in sorted(data, key=keyf):
        e = data[k]
        cells = [k[0], k[1] if k[1] == "float32" else k[1][1:], f"{e['PyTorch']:.2f}"]
        for fmt, _, _ in FORMATS:
            if fmt in e:
                d = e[fmt] - e["PyTorch"]
                worst[fmt] = max(worst[fmt], abs(d))
                cells.append(f"{e[fmt]:.2f} ({d:+.2f})")
            else:
                cells.append("n/a")
        lines.append("| " + " | ".join(cells) + " |")
    summary = ", ".join(f"{f} {w:.2f}%" for f, w in worst.items())
    block_body = "\n".join(lines) + f"\n\nLargest |mean-AP difference| vs PyTorch per format: {summary}.\n"

    if not args.out:
        print(block_body)
        return
    out = Path(args.out)
    text = out.read_text()
    s, e_ = f"<!-- {args.section_marker} START -->", f"<!-- {args.section_marker} END -->"
    block = f"{s}\n{block_body}{e_}"
    pat = re.compile(re.escape(s) + r".*?" + re.escape(e_), re.DOTALL)
    assert pat.search(text), f"add the {s} / {e_} markers to {out} first"
    out.write_text(pat.sub(lambda _: block, text))
    print(f"wrote {len(data)} rows -> {out}")


if __name__ == "__main__":
    main()
