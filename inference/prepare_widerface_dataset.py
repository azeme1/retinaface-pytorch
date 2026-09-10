"""Downloads and lays out the WIDER FACE dataset + the widerface_evaluation
tool this repo's whole eval pipeline (inference/export_*_check.py, and this
project's own private training/eval pipeline) expects on disk -- currently a
fully manual, README-only process (see README.md's "Download the WIDERFACE
Dataset" / "Evaluating RetinaFace on WiderFace Dataset" sections) with no
existing script to reuse; this is that script.

Idempotent: every step checks whether its target already exists (this
machine already has both set up) before downloading/building anything, so
re-running is always safe and near-instant once everything's in place.

Produces:
    data/widerface/
        train/images/<event>/<file>.jpg, train/label.txt   (only with --train)
        val/images/<event>/<file>.jpg, val/wider_val.txt
    widerface_evaluation/
        ground_truth/wider_{easy,medium,hard}_val.mat, wider_face_val.mat
        bbox.cpython-*.so   (compiled from box_overlaps.pyx)

Sources (same ones README.md documents, just automated):
    - Dataset (pre-organized train+val, credited to biubug6 in README.md):
      Google Drive folder id 11UGV3nbVv1x9IC--_tK3Uxf7hA6rlbsS
    - Eval tool: https://github.com/yakhyo/widerface_evaluation (ground_truth
      .mat files ship inside this clone -- no separate download for those)

Usage:
    python inference/prepare_widerface_dataset.py            # val only (eval/export-check workflows)
    python inference/prepare_widerface_dataset.py --train     # also fetch train/ (needed to retrain a backbone)
    python inference/prepare_widerface_dataset.py --force     # re-download/rebuild even if already present
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

RETINA_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = RETINA_DIR / "data" / "widerface"
EVAL_DIR = RETINA_DIR / "widerface_evaluation"

DATASET_GDRIVE_FOLDER_ID = "11UGV3nbVv1x9IC--_tK3Uxf7hA6rlbsS"
EVAL_TOOL_REPO = "https://github.com/yakhyo/widerface_evaluation"


def _run(cmd: list[str], **kwargs) -> None:
    print(f"$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, **kwargs)


def _val_ready() -> bool:
    return (DATA_DIR / "val" / "wider_val.txt").exists() and (DATA_DIR / "val" / "images").is_dir()


def _train_ready() -> bool:
    return (DATA_DIR / "train" / "label.txt").exists() and (DATA_DIR / "train" / "images").is_dir()


def _eval_tool_ready() -> bool:
    gt = EVAL_DIR / "ground_truth"
    required = ["wider_easy_val.mat", "wider_medium_val.mat", "wider_hard_val.mat", "wider_face_val.mat"]
    has_mats = gt.is_dir() and all((gt / f).exists() for f in required)
    has_compiled_bbox = any(EVAL_DIR.glob("bbox*.so")) or any(EVAL_DIR.glob("bbox*.pyd"))
    return has_mats and has_compiled_bbox


def ensure_dataset(want_train: bool, force: bool) -> None:
    need_val = force or not _val_ready()
    need_train = want_train and (force or not _train_ready())
    if not need_val and not need_train:
        print("[dataset] val/ (and train/, if requested) already present -- skipping download")
        return

    try:
        import gdown  # noqa: F401
    except ImportError:
        _run([sys.executable, "-m", "pip", "install", "gdown"])

    DATA_DIR.parent.mkdir(parents=True, exist_ok=True)
    download_dir = DATA_DIR.parent / "_widerface_gdrive_download"
    download_dir.mkdir(exist_ok=True)
    print(f"[dataset] downloading pre-organized WIDER FACE dataset (Google Drive folder "
          f"{DATASET_GDRIVE_FOLDER_ID}) -- this can take a while (train+val images)")
    _run([sys.executable, "-m", "gdown", "--folder", DATASET_GDRIVE_FOLDER_ID, "-O", str(download_dir)])

    # The folder's own internal layout is expected to already match
    # data/widerface/{train,val}/... (that's the whole point of the
    # "pre-organized" bundle) -- move whichever of train/val this call
    # actually needs into place, leaving the other alone if not requested.
    extracted_root = download_dir
    candidates = list(download_dir.rglob("wider_val.txt"))
    if candidates:
        extracted_root = candidates[0].parent.parent  # .../widerface/val/wider_val.txt -> .../widerface

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if need_val:
        src_val = extracted_root / "val"
        assert src_val.is_dir(), f"expected {src_val} after download -- got unexpected layout, inspect {download_dir}"
        dst_val = DATA_DIR / "val"
        if dst_val.exists():
            shutil.rmtree(dst_val)
        shutil.move(str(src_val), str(dst_val))
        print(f"[dataset] val/ ready: {dst_val}")

    if need_train:
        src_train = extracted_root / "train"
        assert src_train.is_dir(), f"expected {src_train} after download -- got unexpected layout, inspect {download_dir}"
        dst_train = DATA_DIR / "train"
        if dst_train.exists():
            shutil.rmtree(dst_train)
        shutil.move(str(src_train), str(dst_train))
        print(f"[dataset] train/ ready: {dst_train}")

    shutil.rmtree(download_dir, ignore_errors=True)


def ensure_eval_tool(force: bool) -> None:
    if not force and _eval_tool_ready():
        print("[eval_tool] widerface_evaluation/ already built -- skipping")
        return

    if not (EVAL_DIR / ".git").exists():
        _run(["git", "clone", EVAL_TOOL_REPO, str(EVAL_DIR)])
    else:
        print(f"[eval_tool] {EVAL_DIR} already cloned, reusing")

    _run([sys.executable, "setup.py", "build_ext", "--inplace"], cwd=str(EVAL_DIR))

    gt = EVAL_DIR / "ground_truth"
    required = ["wider_easy_val.mat", "wider_medium_val.mat", "wider_hard_val.mat", "wider_face_val.mat"]
    missing = [f for f in required if not (gt / f).exists()]
    assert not missing, f"widerface_evaluation clone is missing ground_truth files: {missing}"
    print(f"[eval_tool] ready: {EVAL_DIR}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train", action="store_true", help="also fetch data/widerface/train/ (needed to retrain a "
                                                          "backbone from scratch; not needed for eval/export-check)")
    p.add_argument("--force", action="store_true", help="re-download/rebuild even if already present")
    p.add_argument("--skip-eval-tool", action="store_true", help="skip cloning/building widerface_evaluation")
    args = p.parse_args()

    ensure_dataset(want_train=args.train, force=args.force)
    if not args.skip_eval_tool:
        ensure_eval_tool(force=args.force)

    n_val_images = sum(1 for _ in (DATA_DIR / "val" / "images").rglob("*.jpg")) if _val_ready() else 0
    n_val_list = len((DATA_DIR / "val" / "wider_val.txt").read_text().split()) if _val_ready() else 0
    print(f"\n[done] val/: {n_val_images} images on disk, {n_val_list} entries in wider_val.txt "
          f"(expect 3226 images, 3226 entries)")
    if args.train:
        n_train_images = sum(1 for _ in (DATA_DIR / "train" / "images").rglob("*.jpg")) if _train_ready() else 0
        print(f"[done] train/: {n_train_images} images on disk (expect 12880)")
    print(f"[done] eval tool: {'ready' if _eval_tool_ready() else 'NOT ready -- see errors above'}")


if __name__ == "__main__":
    main()
