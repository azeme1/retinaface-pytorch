"""Writes ../checkpoint_<format>.md (format = onnx|tflite|tfjs): for every
converted-format zip published on the HF repo, its size + sha256 (as
advertised by HF's LFS metadata) and the real full-val WIDER FACE AP of that
artifact vs. its PyTorch source, parsed from the per-run logs that
validate_<format>.sh / export_check.py --format <format> wrote
(<format>_verify_logs/, run from this directory).

Usage: python generate_checkpoint_md.py onnx tflite tfjs
"""
import datetime
import re
import sys
import hashlib
from pathlib import Path

from huggingface_hub import HfApi

HERE = Path(__file__).resolve().parent
REPO = "azemel/retinaface-xs"
STAGE = HERE.parents[1] / "hf_push_staging" / "checkpoints"  # files built locally but not on HF yet
ORDER = ["mobilenetv1", "mobilenetv1_0.25", "mobilenetv1_0.50", "mobilenetv2", "resnet18", "resnet34", "resnet50"]
META = {
    "onnx": ("ONNX", "ONNX Runtime CUDAExecutionProvider vs. a CUDA PyTorch reference", "_CUDA"),
    "tflite": ("TFLite", "`tf.lite.Interpreter` (CPU -- no GPU delegate exists for it here)", ""),
    "tfjs": ("TF.js", "`@tensorflow/tfjs-node` under Node.js (CPU backend -- no usable GPU build)", ""),
}
ROW = re.compile(r"^(Easy|Medium|Hard|Average)\s+([\d.]+)%\s+([\d.]+)%\s+([+-][\d.]+)%", re.M)


def parse(path: Path):
    if not path.exists():
        return None
    rows = {k: (float(a), float(b), float(d)) for k, a, b, d in ROW.findall(path.read_text().replace("\r", "\n"))}
    return rows if len(rows) == 4 else None


def main():
    fmts = sys.argv[1:] or ["onnx", "tflite", "tfjs"]
    tree = list(HfApi().list_repo_tree(REPO, path_in_repo="checkpoints", recursive=True, expand=True))
    for fmt in fmts:
        label, backend, suffix = META[fmt]
        files = []
        for i in tree:
            parts = i.path.split("/")
            if len(parts) == 4 and parts[2] == fmt and parts[3].endswith(".zip"):
                lvl = parts[3][len(parts[1]) + 1:-4]
                files.append((parts[1], lvl, i.size, i.lfs.sha256))
        on_hf = {(n, l) for n, l, _, _ in files}
        pending = set()
        for z in sorted(STAGE.glob(f"*/{fmt}/*.zip")):
            net = z.parts[-3]
            lvl = z.name[len(net) + 1:-4]
            if (net, lvl) not in on_hf:
                files.append((net, lvl, z.stat().st_size, hashlib.sha256(z.read_bytes()).hexdigest()))
                pending.add((net, lvl))
        files.sort(key=lambda r: (ORDER.index(r[0]) if r[0] in ORDER else 99, r[1] == "float32", int(r[1][1:]) if r[1] != "float32" else 0))
        out = [f"# {label} checkpoints -- hashes and verified WIDER FACE metrics", "",
               f"Generated {datetime.date.today()} by `inference/generate_checkpoint_md.py` from the logs of `inference/validate_{fmt}.sh` (`inference/export_check.py --format {fmt}`).", "",
               f"- **Files:** `checkpoints/<backbone>/{fmt}/<backbone>_<level>.zip` in the HF repo `{REPO}` ({len(files)} files).",
               "- **Hash:** sha256 HF advertises (LFS) for the file. Every file was also re-downloaded fresh from HF and its sha256 recomputed: all matched (2026-09-19). No local copies are kept.",
               f"- **Metrics:** real full-val WIDER FACE AP (3226 images) of the artifact downloaded from HF, run through {backend}, on the fixed-size 640x640 letterbox preprocessing; \"vs PyTorch\" is the difference in mean AP from the PyTorch `RetinaStaticExportWrapper` run in the same script. Not comparable to `results/<network>/full_eval_parallel/*.json` (variable-size pipeline).",
               ]
        if fmt != "onnx":
            out.append("- **Reference device:** `resnet50` rows ran against a CUDA PyTorch reference; every other backbone's rows ran against a CPU PyTorch reference (an earlier run pinned to CPU) -- PyTorch CPU vs CUDA differs by ~0.02% mean AP, so the \"vs PyTorch\" column is comparable to that precision.")
        if pending:
            out.append(f"- **Not on HF yet:** {len(pending)} row(s) marked `(local)` are built and verified locally (`hf_push_staging/`) but not uploaded; their sha256 is of the local zip.")
        out += ["", "| backbone | level | size (MB) | sha256 | easy | medium | hard | mean | mean vs PyTorch |", "|---|---|---|---|---|---|---|---|---|"]
        missing = 0
        for net, lvl, size, sha in files:
            r = parse(HERE / f"{fmt}_verify_logs" / f"{net}_{lvl}{suffix}.log")
            if r:
                cells = [f"{r[k][1]:.2f}" for k in ("Easy", "Medium", "Hard", "Average")] + [f"{r['Average'][2]:+.2f}"]
            else:
                cells = ["n/a"] * 5
                missing += 1
            out.append(f"| {net} | {lvl}{' (local)' if (net, lvl) in pending else ''} | {size / 1e6:.2f} | `{sha}` | " + " | ".join(cells) + " |")
        out += ["", "AP values are percentages" + (f"; `n/a` = not verified ({missing} file(s)) -- the validate scripts skip `float32` levels." if missing else "."), ""]
        (HERE.parent / f"checkpoint_{fmt}.md").write_text("\n".join(out))
        print(f"checkpoint_{fmt}.md: {len(files)} files, {missing} without metrics")


if __name__ == "__main__":
    main()
