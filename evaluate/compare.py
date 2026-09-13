"""Compares one or more backbone checkpoints against one or more formats
(pytorch, onnx, coreml, tflite) on the real WIDER FACE val set, for one or
more compression levels, and writes every (level, format) result as a row
to a CSV.

--format defaults to just ["pytorch"] -- the real, variable-size,
per-image-PriorBox, aspect-preserving pipeline (widerface_eval.py's own,
the same methodology evaluate/run_parallel.py uses), loading the PUBLIC
plain+clusters HF checkpoint (export_common.load_plain_with_clusters_from_hf)
rather than a private raw-training-checkpoint reconstruction. Pass two or
more --format values (e.g. --format pytorch onnx coreml) to also get
ONNX/CoreML/TFLite numbers in the same run, each obtained by shelling out to
inference/export_check.py -- one canonical implementation of the fixed-size-
canvas comparison, not a second copy of that preprocessing/postprocessing
logic. A non-pytorch format's row also carries the fixed-canvas PyTorch
mean (pytorch_fixed_canvas_mean) alongside its own, since that's what
export_check.py already computes as its reference point -- comparing it
against the "pytorch" row's real mean (if requested) separates the cost of
the fixed-size export canvas itself from the cost of the format conversion
on top of it, without needing two separate rows to do it.

Usage:
    python compare.py --network mobilenetv1 --levels c2 c7 c64 c256
    python compare.py --network mobilenetv1 --levels c7 --format pytorch onnx coreml \\
        --compute-units CPU_ONLY --out mobilenetv1_c7.csv
    python compare.py --network resnet34 --levels c2 c4 c128 c256 --format onnx tflite \\
        --n-images 200   # exact count -- quick spot check instead of the full 3226-image val set
    python compare.py --network resnet34 --levels c256 --format onnx --n-images 0.1  # 10% of the val set
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
import tempfile
import time
import zipfile
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
from export_common import download_hf_artifact, load_plain_with_clusters_from_zip  # noqa: E402

DATASET_FOLDER = str(_RETINA_DIR / "data/widerface/val/images/")
VAL_LIST = str(_RETINA_DIR / "data/widerface/val/wider_val.txt")
GT_DIR = str(_RETINA_DIR / "widerface_evaluation/ground_truth")

# Each print_ap_report() call in export_check.py prints exactly this
# 4-line block (Easy/Medium/Hard/Average); a full run with a runnable
# format prints two of them, in order: PyTorch (fixed-canvas), then format.
_METRIC_BLOCK_RE = re.compile(
    r"Easy:\s*([\d.]+)%\s*\n\s*Medium:\s*([\d.]+)%\s*\n\s*Hard:\s*([\d.]+)%\s*\n\s*Average:\s*([\d.]+)%"
)

CSV_FIELDS = ["network", "level", "format", "easy", "medium", "hard", "mean",
              "pytorch_fixed_canvas_mean", "time_seconds", "source"]


def n_images_type(s: str) -> int | float:
    """--n-images is smart-typed: an integer string ("200") means an exact
    image count, a float string ("0.1") means a ratio of the full val set
    (resolved to a count once the true total is known, see resolve_n_images
    below) -- and not passing --n-images at all (default None) means the
    full dataset. No -1-as-sentinel hack: the type of the value you pass is
    the signal."""
    try:
        return int(s)
    except ValueError:
        return float(s)


def resolve_n_images(n_images: int | float | None, total: int) -> int | None:
    """None (not passed) -> None (we.evaluate_model()'s own "use everything"
    value). An int -> itself, clamped to total. A float -> that fraction of
    total, rounded."""
    if n_images is None:
        return None
    if isinstance(n_images, float):
        return max(1, min(total, round(n_images * total)))
    return max(1, min(total, n_images))


def convert_to_format(fmt: str, network: str, pytorch_zip: Path, image_size: int, out_dir: Path) -> Path:
    """Converts the given plain+clusters pytorch checkpoint to fmt, returns
    the local artifact path (a raw .onnx/.mlpackage/.tflite, or a .zip --
    both are accepted by export_check.py's --artifact)."""
    py = sys.executable
    if fmt == "onnx":
        subprocess.run([py, str(_INFER_DIR / "export_onnx.py"), "--network", network,
                         "--checkpoint", str(pytorch_zip), "--image-size", str(image_size),
                         "--out-dir", str(out_dir)], check=True)
        return next(out_dir.glob("*.zip"))

    if fmt == "coreml":
        subprocess.run([py, str(_INFER_DIR / "export_coreml.py"), "--network", network,
                         "--checkpoint", str(pytorch_zip), "--image-size", str(image_size),
                         "--out-dir", str(out_dir)], check=True)
        return next(out_dir.glob("*.mlpackage"))

    if fmt == "tflite":
        # Two-stage, matching this repo's own tflite export pipeline:
        # --no-palettize dense ONNX (onnx2tf mis-converts grouped/depthwise
        # convs following this repo's Gather-based palettization chain, see
        # export_onnx.py's --no-palettize --help), then onnx -> tflite.
        onnx_dir = out_dir / "onnx_stage"
        onnx_dir.mkdir()
        subprocess.run([py, str(_INFER_DIR / "export_onnx.py"), "--network", network,
                         "--checkpoint", str(pytorch_zip), "--image-size", str(image_size),
                         "--out-dir", str(onnx_dir), "--no-palettize"], check=True)
        onnx_zip = next(onnx_dir.glob("*.zip"))
        with zipfile.ZipFile(onnx_zip) as zf:
            zf.extract("model.onnx", onnx_dir)
        tflite_dir = out_dir / "tflite_stage"
        subprocess.run([py, str(_INFER_DIR / "export_tflite.py"),
                         "--onnx", str(onnx_dir / "model.onnx"), "--out-dir", str(tflite_dir)], check=True)
        return tflite_dir / "model_dynamic_range_quant.tflite"

    raise ValueError(fmt)


def resolve_artifact(fmt: str, network: str, level: str, pytorch_zip: Path, image_size: int,
                      hf_repo: str, hf_token: str | None, convert: bool, tmp_dir: Path) -> Path:
    if not convert:
        try:
            return Path(download_hf_artifact(hf_repo, network, fmt, level, token=hf_token))
        except Exception as e:  # noqa: BLE001 -- fall through to local conversion
            print(f"  no HF {fmt} artifact ({type(e).__name__}: {e}) -- converting locally", flush=True)
    return convert_to_format(fmt, network, pytorch_zip, image_size, tmp_dir)


def run_pytorch_row(network: str, level: str, pytorch_zip: Path, cfg: dict, device: torch.device,
                     n_images: int | None, num_workers: int, num_threads: int, pred_dir: Path) -> dict:
    """The real, variable-size, aspect-preserving full-val AP -- the same
    methodology evaluate/run_parallel.py uses, on the PUBLIC HF checkpoint.
    n_images is already resolved to an absolute count or None (full);
    see resolve_n_images."""
    model, _ = load_plain_with_clusters_from_zip(cfg, str(pytorch_zip))
    model = model.to(device)
    t0 = time.time()
    aps = we.evaluate_model(model, cfg, device, DATASET_FOLDER, VAL_LIST, GT_DIR,
                             pred_dir=str(pred_dir), sample_size=n_images,
                             num_workers=num_workers, log_progress=True, eval_num_workers=num_threads)
    elapsed = time.time() - t0
    # *100: every other row's numbers come from export_check.py's printed
    # percentages (0-100) -- keep this column consistently in the same
    # units, not we.mean_ap()'s raw 0-1 fraction.
    return {
        "network": network, "level": level, "format": "pytorch",
        "easy": aps["easy"] * 100, "medium": aps["medium"] * 100, "hard": aps["hard"] * 100,
        "mean": we.mean_ap(aps) * 100,
        "pytorch_fixed_canvas_mean": "", "time_seconds": f"{elapsed:.1f}", "source": "real (variable-size)",
    }


def run_format_row(fmt: str, network: str, level: str, pytorch_zip: Path, image_size: int,
                    n_images: int | None, hf_repo: str, hf_token: str | None, convert: bool,
                    onnx_provider: str | None, compute_units: str | None) -> dict:
    """n_images is already resolved to an absolute count or None (full);
    see resolve_n_images."""
    with tempfile.TemporaryDirectory() as tmp:
        artifact_path = resolve_artifact(fmt, network, level, pytorch_zip, image_size,
                                          hf_repo, hf_token, convert, Path(tmp))
        cmd = [sys.executable, str(_INFER_DIR / "export_check.py"), "--format", fmt,
               "--network", network, "--checkpoint", str(pytorch_zip), "--artifact", str(artifact_path),
               "--image-size", str(image_size)]
        if n_images is not None:
            cmd += ["--n-images", str(n_images)]
        if fmt == "onnx":
            cmd += ["--onnx-provider", onnx_provider or "CPU"]
        if fmt == "coreml":
            cmd += ["--compute-units", compute_units or "CPU_ONLY"]

        t0 = time.time()
        r = subprocess.run(cmd, cwd=str(_INFER_DIR), capture_output=True, text=True)
        elapsed = time.time() - t0
        out = r.stdout + "\n" + r.stderr
        if r.returncode != 0:
            raise RuntimeError(f"export_check.py failed (exit {r.returncode}):\n{out[-3000:]}")

        blocks = _METRIC_BLOCK_RE.findall(out)
        if len(blocks) < 2:
            # can_run was False (e.g. CoreML .predict() unavailable on this
            # platform) -- only the PyTorch block printed.
            assert len(blocks) == 1, f"expected 1 or 2 metric blocks, got {len(blocks)}:\n{out[-2000:]}"
            pt_easy, pt_medium, pt_hard, pt_mean = (float(x) for x in blocks[0])
            print(f"  [warning] {fmt} could not actually run on this platform -- see log", flush=True)
            return {
                "network": network, "level": level, "format": fmt,
                "easy": "", "medium": "", "hard": "", "mean": "",
                "pytorch_fixed_canvas_mean": pt_mean, "time_seconds": f"{elapsed:.1f}",
                "source": "NOT RUN on this platform",
            }
        (pt_easy, pt_medium, pt_hard, pt_mean), (f_easy, f_medium, f_hard, f_mean) = blocks[0], blocks[1]
        return {
            "network": network, "level": level, "format": fmt,
            "easy": f_easy, "medium": f_medium, "hard": f_hard, "mean": f_mean,
            "pytorch_fixed_canvas_mean": pt_mean, "time_seconds": f"{elapsed:.1f}",
            "source": "fixed-canvas (letterboxed export graph)",
        }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--network", required=True)
    p.add_argument("--levels", nargs="+", required=True, help="e.g. c2 c7 c64 c256 float32")
    p.add_argument("--format", nargs="+", default=["pytorch"], choices=["pytorch", "onnx", "coreml", "tflite"],
                   help="one or more of pytorch/onnx/coreml/tflite -- default is just pytorch "
                        "(the real full-val baseline); pass more to compare them in the same run")
    p.add_argument("--hf-repo", default="azemel/retinaface-xs")
    p.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    p.add_argument("--image-size", type=int, default=640)
    p.add_argument("--n-images", type=n_images_type, default=None,
                    help="WIDER FACE val images to sample -- smart-typed: omit for the FULL val set "
                         "(3226 images, the only way to get an AP comparable across runs), pass an "
                         "integer (e.g. 200) for an exact count, or a float (e.g. 0.1) for that "
                         "fraction of the full set -- either way a quicker, noisier spot check")
    p.add_argument("--num-workers", type=int, default=2,
                    help="ThreadPoolExecutor width for widerface_eval.py's per-image loop, pytorch side")
    p.add_argument("--num-threads", type=int, default=4, help="cap on torch's own intra-op CPU thread pool")
    p.add_argument("--device", default=None, choices=["cuda", "mps", "cpu"],
                    help="force the pytorch side onto this device; None auto-picks the best "
                         "available accelerator (cuda, then mps, then cpu)")
    p.add_argument("--onnx-provider", default=None, choices=["CPU", "CUDA"], help="--format onnx only")
    p.add_argument("--compute-units", default=None,
                    choices=["CPU_ONLY", "CPU_AND_GPU", "CPU_AND_NE", "ALL"], help="--format coreml only")
    p.add_argument("--convert", action="store_true",
                    help="always convert locally instead of trying an existing HF format artifact first")
    p.add_argument("--out", default=None, help="CSV path -- default: <network>_compare.csv")
    p.add_argument("--pred-dir-root", default=None,
                    help="where the pytorch side's per-image prediction .txt files go; "
                         "default: results/<network>/compare_pred")
    args = p.parse_args()

    out_path = Path(args.out) if args.out else Path(f"{args.network}_compare.csv")
    pred_dir_root = Path(args.pred_dir_root) if args.pred_dir_root else \
        _RETINA_DIR / "results" / args.network / "compare_pred"
    pred_dir_root.mkdir(parents=True, exist_ok=True)

    torch.set_num_threads(args.num_threads)
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    cfg = dict(get_config(args.network))

    total_images = len(Path(VAL_LIST).read_text().split())
    n_images = resolve_n_images(args.n_images, total_images)
    if n_images is not None:
        print(f"[compare] sampling {n_images}/{total_images} val images "
              f"(--n-images {args.n_images!r})", flush=True)

    rows = []
    for level in args.levels:
        print(f"\n########## {args.network} {level} ##########", flush=True)
        pytorch_zip = Path(download_hf_artifact(args.hf_repo, args.network, "pytorch", level, token=args.hf_token))
        for fmt in args.format:
            print(f"-- {fmt} --", flush=True)
            if fmt == "pytorch":
                row = run_pytorch_row(args.network, level, pytorch_zip, cfg, device, n_images,
                                       args.num_workers, args.num_threads, pred_dir_root / f"{level}_pytorch")
            else:
                row = run_format_row(fmt, args.network, level, pytorch_zip, args.image_size, n_images,
                                      args.hf_repo, args.hf_token, args.convert,
                                      args.onnx_provider, args.compute_units)
            rows.append(row)
            mean_str = f"{float(row['mean']) if row['mean'] != '' else float('nan'):.2f}%" if row["mean"] != "" else "n/a"
            print(f"  {level}/{fmt}: mean={mean_str}", flush=True)

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} row(s) -> {out_path}")


if __name__ == "__main__":
    main()
