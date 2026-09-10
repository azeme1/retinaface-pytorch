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
    python inference/prepare_widerface_dataset.py              # fetches both train/ and val/
    python inference/prepare_widerface_dataset.py --val-only    # skip train/ (eval/export-check workflows only)
    python inference/prepare_widerface_dataset.py --force       # re-download/rebuild even if already present
"""
import argparse
import shutil
import subprocess
import sys
import time
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


def _print_tree(root: Path, max_entries: int = 300) -> None:
    print(f"[dataset] directory tree under {root}:")
    count = 0
    for p in sorted(root.rglob("*")):
        print(f"  {p.relative_to(root)}{'/' if p.is_dir() else ''}")
        count += 1
        if count >= max_entries:
            print(f"  ... (truncated after {max_entries} entries)")
            break


def _find_split_dir(extracted_root: Path, download_dir: Path, split: str) -> Path:
    """Locates the downloaded split's directory (val or train) even when
    gdown's --folder mode doesn't reproduce the exact single-top-level-
    "widerface"-folder nesting this script originally assumed -- gdown's
    --folder layout has been observed to vary (extra nesting, a top-level
    folder named after the Drive folder itself, or partially-failed
    sub-downloads on a rate-limited pull) depending on machine/network.
    Tries, in order: the originally assumed path, a directory literally
    named `split` anywhere in the download, and the official WIDER FACE
    zip naming (WIDER_val/WIDER_train) anywhere in the download -- accepts
    the first candidate that at least has an images/ subdirectory (the
    part that actually matters; a missing wider_val.txt/label.txt list
    file only warns, since eval/training need that too but it doesn't
    block locating the directory itself)."""
    marker = "wider_val.txt" if split == "val" else "label.txt"
    candidates = [extracted_root / split]
    candidates += sorted(d for d in download_dir.rglob(split) if d.is_dir())
    candidates += sorted(d for d in download_dir.rglob(f"WIDER_{split}") if d.is_dir())
    for c in candidates:
        if c.is_dir() and (c / "images").is_dir():
            if not (c / marker).exists():
                print(f"[dataset] warning: {c} has images/ but no {marker} -- eval/training will need that "
                      f"list file too; check the download or the README's manual steps for it")
            return c
    _print_tree(download_dir)
    raise AssertionError(
        f"couldn't find a usable {split}/ directory (one containing an images/ subfolder) after download -- "
        f"see the directory tree printed above and compare against what README.md's manual steps expect, "
        f"or inspect {download_dir} yourself. A common cause is gdown's --folder mode partially failing on a "
        f"large/rate-limited folder -- rerun with --force, or download+extract the folder manually into "
        f"{download_dir} and rerun."
    )


def _attempt_gdown_folder_download(folder_id: str, download_dir: Path) -> bool:
    """One gdown --folder attempt, streaming its output live. Returns
    whether it actually produced files -- gdown's own known failure mode
    here ("Retrieving folder contents" / "Failed to retrieve folder
    contents", e.g. Drive rate limiting, an outdated gdown build, or a
    folder too large to list unauthenticated) exits 0 despite failing, so
    a non-empty download_dir is the only reliable success signal."""
    proc = subprocess.Popen(
        [sys.executable, "-m", "gdown", "--folder", folder_id, "-O", str(download_dir)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
    )
    output = []
    for line in proc.stdout:
        print(line, end="", flush=True)
        output.append(line)
    proc.wait()
    return proc.returncode == 0 and "Failed to retrieve folder contents" not in "".join(output) \
        and any(download_dir.iterdir())


def _download_gdrive_folder_with_retries(folder_id: str, download_dir: Path,
                                          max_attempts: int = 4, backoff_seconds: float = 10.0) -> None:
    """Fully automatic: retries the flaky gdown --folder listing itself --
    upgrades gdown once (the failure is often gdown falling behind a
    changed Drive page format) then keeps retrying with backoff (the
    failure is also often just transient Drive rate limiting), no
    intervention needed unless every attempt genuinely exhausts itself."""
    upgraded = False
    for attempt in range(1, max_attempts + 1):
        print(f"[dataset] gdown attempt {attempt}/{max_attempts}...")
        if _attempt_gdown_folder_download(folder_id, download_dir):
            return
        for f in download_dir.iterdir():
            shutil.rmtree(f) if f.is_dir() else f.unlink()  # clear a partial/empty attempt before retrying
        if not upgraded:
            print("[dataset] gdown failed to list the folder -- upgrading gdown and retrying "
                  "(this failure mode is usually gdown falling behind a Drive page-format change)")
            _run([sys.executable, "-m", "pip", "install", "-U", "gdown"])
            upgraded = True
        elif attempt < max_attempts:
            print(f"[dataset] still failing -- Drive rate limiting is often transient, "
                  f"waiting {backoff_seconds:.0f}s before retrying...")
            time.sleep(backoff_seconds)

    raise RuntimeError(
        f"gdown could not retrieve the Google Drive folder after {max_attempts} automatic attempts "
        f"(including a gdown upgrade) -- see its output above. This particular folder/network combination "
        f"may need a manual download: open "
        f"https://drive.google.com/drive/folders/{folder_id} in a browser, download it, extract it into "
        f"{download_dir}, and rerun this script (it accepts several sub-layouts once files are actually "
        f"there -- see _find_split_dir)."
    )


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
    _download_gdrive_folder_with_retries(DATASET_GDRIVE_FOLDER_ID, download_dir)

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
        src_val = _find_split_dir(extracted_root, download_dir, "val")
        dst_val = DATA_DIR / "val"
        if dst_val.exists():
            shutil.rmtree(dst_val)
        shutil.move(str(src_val), str(dst_val))
        print(f"[dataset] val/ ready: {dst_val}")

    if need_train:
        src_train = _find_split_dir(extracted_root, download_dir, "train")
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
    p.add_argument("--val-only", action="store_true",
                    help="skip data/widerface/train/ -- by default both splits are fetched, since gdown "
                         "downloads the whole shared folder (train+val together) in one shot regardless, so "
                         "skipping train afterward only discards already-downloaded data, not bandwidth/time")
    p.add_argument("--force", action="store_true", help="re-download/rebuild even if already present")
    p.add_argument("--skip-eval-tool", action="store_true", help="skip cloning/building widerface_evaluation")
    args = p.parse_args()
    want_train = not args.val_only

    ensure_dataset(want_train=want_train, force=args.force)
    if not args.skip_eval_tool:
        ensure_eval_tool(force=args.force)

    n_val_images = sum(1 for _ in (DATA_DIR / "val" / "images").rglob("*.jpg")) if _val_ready() else 0
    n_val_list = len((DATA_DIR / "val" / "wider_val.txt").read_text().split()) if _val_ready() else 0
    print(f"\n[done] val/: {n_val_images} images on disk, {n_val_list} entries in wider_val.txt "
          f"(expect 3226 images, 3226 entries)")
    if want_train:
        n_train_images = sum(1 for _ in (DATA_DIR / "train" / "images").rglob("*.jpg")) if _train_ready() else 0
        print(f"[done] train/: {n_train_images} images on disk (expect 12880)")
    print(f"[done] eval tool: {'ready' if _eval_tool_ready() else 'NOT ready -- see errors above'}")


if __name__ == "__main__":
    main()
