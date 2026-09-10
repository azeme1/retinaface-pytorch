"""Downloads and lays out the WIDER FACE dataset + the widerface_evaluation
tool this repo's whole eval pipeline (inference/export_*_check.py, and this
project's own private training/eval pipeline) expects on disk -- currently a
fully manual, README-only process (see README.md's "Download the WIDERFACE
Dataset" / "Evaluating RetinaFace on WiderFace Dataset" sections) with no
existing script to reuse; this is that script.

Downloads from the CUHK-CSE/wider_face Hugging Face dataset repo -- a
re-host of the official http://shuoyang1213.me/WIDERFACE/ distribution
(WIDER_train.zip, WIDER_val.zip, wider_face_split.zip) on HF's own CDN via
plain huggingface_hub file downloads (resumable, no folder-listing step to
break). This replaced an earlier gdown-against-a-shared-Google-Drive-folder
approach that proved unreliable in practice (Drive's folder-listing API
rate-limits or breaks outright depending on gdown version/network -- see
git history for the retry/upgrade logic that was needed to work around it,
now unnecessary).

IMPORTANT caveat this rewrite introduces: CUHK-CSE/wider_face re-hosts the
OFFICIAL WIDER FACE distribution -- images + bounding boxes only. It does
NOT include the 5-point facial landmark annotations RetinaFace's own
label.txt format was designed around for training (those are a separate
contribution, biubug6's retinaface_gt_v1.1 bundle, not re-hosted on HF).
train/label.txt is generated here from the official bounding-box-only
wider_face_train_bbx_gt.txt with every face's landmarks marked invalid
(-1) -- utils/dataset.py's WiderFaceDetection loader already handles a
per-face invalid-landmark marker (see its __getitem__: `1 if label[4] >= 0
else -1`), so training RUNS on this data, but with zero landmark
supervision throughout the whole run -- NOT equivalent to a checkpoint
trained on the original biubug6 annotations. Fine for eval/export-check
(which never touches landmark ground truth here -- WIDER FACE's own AP
scoring is box-only) or a bbox-only ablation; NOT fine for reproducing a
real landmark-capable checkpoint from scratch.

Idempotent: every step checks whether its target already exists (this
machine already has both set up) before downloading/building anything, so
re-running is always safe and near-instant once everything's in place.

Produces:
    data/widerface/
        train/images/<event>/<file>.jpg, train/label.txt   (only with both splits, see --val-only)
        val/images/<event>/<file>.jpg, val/wider_val.txt
    widerface_evaluation/
        ground_truth/wider_{easy,medium,hard}_val.mat, wider_face_val.mat
        bbox.cpython-*.so   (compiled from box_overlaps.pyx)

Sources:
    - Dataset: https://huggingface.co/datasets/CUHK-CSE/wider_face (re-host
      of the official http://shuoyang1213.me/WIDERFACE/ distribution)
    - Eval tool: https://github.com/yakhyo/widerface_evaluation (ground_truth
      .mat files ship inside this clone -- no separate download for those)

Usage:
    python inference/prepare_widerface_dataset.py              # fetches both train/ and val/
    python inference/prepare_widerface_dataset.py --val-only    # skip train/ (eval/export-check workflows only)
    python inference/prepare_widerface_dataset.py --force       # re-download/rebuild even if already present
"""
import argparse
import os
import subprocess
import sys
import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download

RETINA_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = RETINA_DIR / "data" / "widerface"
EVAL_DIR = RETINA_DIR / "widerface_evaluation"

DATASET_REPO_ID = "CUHK-CSE/wider_face"
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


def _hf_dataset_file(filename: str) -> Path:
    """Downloads one file from CUHK-CSE/wider_face via huggingface_hub's
    own cache (a second call for the same file is instant, no re-download).
    HF_TOKEN is optional -- this dataset is public -- but honored if set,
    to avoid anonymous rate limits on a slow/shared connection."""
    return Path(hf_hub_download(repo_id=DATASET_REPO_ID, repo_type="dataset", filename=filename,
                                 token=os.environ.get("HF_TOKEN")))


def _extract_split_images(zip_path: Path, wider_folder_name: str, dst_split_dir: Path) -> None:
    """Extracts only the `<wider_folder_name>/images/...` member tree from
    an official WIDER_{train,val}.zip (confirmed via direct inspection:
    that's the exact top-level layout CUHK-CSE/wider_face's zips use)
    straight into dst_split_dir/images -- no intermediate extract-then-move
    step needed since the expected internal layout is already known."""
    prefix = f"{wider_folder_name}/images/"
    with zipfile.ZipFile(zip_path) as zf:
        members = [m for m in zf.namelist() if m.startswith(prefix) and not m.endswith("/")]
        assert members, f"{zip_path} has no members under {prefix} -- unexpected zip layout"
        for member in members:
            rel = member[len(prefix):]
            target = dst_split_dir / "images" / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as src, open(target, "wb") as dst:
                dst.write(src.read())


def _parse_bbx_gt(text: str) -> list[tuple[str, list[list[float]]]]:
    """Parses the official wider_face_{train,val}_bbx_gt.txt format:
    filename line, face-count line, then that many
    'x1 y1 w h blur expression illumination invalid occlusion pose' lines
    -- except when count is 0, where the file still has exactly ONE dummy
    line afterward regardless (confirmed by direct inspection: 4 such
    cases in the real train file, 0 in val) -- a well-known WIDER FACE
    annotation-file quirk. Returns [(relative_image_path, [[x1,y1,w,h], ...]), ...],
    skipping the dummy line's contents for count==0 entries."""
    lines = [ln for ln in text.splitlines() if ln.strip() != ""]
    i = 0
    entries = []
    while i < len(lines):
        fname = lines[i].strip()
        i += 1
        count = int(lines[i].strip())
        i += 1
        n_lines_to_consume = count if count > 0 else 1
        boxes = []
        for _ in range(n_lines_to_consume):
            parts = lines[i].split()
            i += 1
            if count > 0:
                boxes.append([float(x) for x in parts[:4]])  # x1, y1, w, h
        entries.append((fname, boxes))
    return entries


def _write_wider_val_list(entries: list[tuple[str, list]], out_path: Path) -> None:
    """widerface_eval.py's own convention: one leading-'/' relative path
    per line (e.g. "/0--Parade/0_Parade_marchingband_1_465.jpg")."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(f"/{fname}\n" for fname, _ in entries))


def _write_label_txt(entries: list[tuple[str, list]], out_path: Path) -> None:
    """RetinaFace's own label.txt format (see utils/dataset.py's
    WiderFaceDetection._parse_labels): '# <relative_path>' header lines
    followed by one 'x1 y1 w h <15 landmark values>' line per face --
    landmarks are all -1 here (see module docstring's caveat: this source
    has no real landmark annotations). Images with zero faces are omitted
    entirely rather than emitted as an empty-body '#' entry -- the loader's
    own flush-on-next-'#' logic only appends a completed image's boxes when
    `if labels:` is true, so a genuinely empty entry would desync
    self.image_paths against self.words for every image after it."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    skipped = 0
    for fname, boxes in entries:
        if not boxes:
            skipped += 1
            continue
        lines.append(f"# {fname}")
        invalid_landmarks = " ".join(["-1"] * 15)
        for x1, y1, w, h in boxes:
            lines.append(f"{x1} {y1} {w} {h} {invalid_landmarks}")
    out_path.write_text("\n".join(lines) + "\n")
    if skipped:
        print(f"[dataset] label.txt: skipped {skipped} zero-face image(s) (see _write_label_txt's docstring)")


def ensure_dataset(want_train: bool, force: bool) -> None:
    need_val = force or not _val_ready()
    need_train = want_train and (force or not _train_ready())
    if not need_val and not need_train:
        print("[dataset] val/ (and train/, if requested) already present -- skipping download")
        return

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[dataset] downloading annotations from Hugging Face ({DATASET_REPO_ID})...")
    split_zip = _hf_dataset_file("data/wider_face_split.zip")
    with zipfile.ZipFile(split_zip) as zf:
        val_bbx_text = zf.read("wider_face_split/wider_face_val_bbx_gt.txt").decode()
        train_bbx_text = zf.read("wider_face_split/wider_face_train_bbx_gt.txt").decode() if need_train else None

    if need_val:
        print(f"[dataset] downloading WIDER_val.zip from Hugging Face ({DATASET_REPO_ID}) -- "
              f"this can take a while (~360 MB)...")
        val_zip = _hf_dataset_file("data/WIDER_val.zip")
        dst_val = DATA_DIR / "val"
        _extract_split_images(val_zip, "WIDER_val", dst_val)
        val_entries = _parse_bbx_gt(val_bbx_text)
        _write_wider_val_list(val_entries, dst_val / "wider_val.txt")
        print(f"[dataset] val/ ready: {dst_val} ({len(val_entries)} images)")

    if need_train:
        print(f"[dataset] downloading WIDER_train.zip from Hugging Face ({DATASET_REPO_ID}) -- "
              f"this can take a while (~1.4 GB)...")
        train_zip = _hf_dataset_file("data/WIDER_train.zip")
        dst_train = DATA_DIR / "train"
        _extract_split_images(train_zip, "WIDER_train", dst_train)
        train_entries = _parse_bbx_gt(train_bbx_text)
        _write_label_txt(train_entries, dst_train / "label.txt")
        print(f"[dataset] train/ ready: {dst_train} ({len(train_entries)} images) -- "
              f"WARNING: landmarks are all-invalid placeholders, see this module's own docstring")


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
                    help="skip data/widerface/train/ -- by default both splits are fetched. Note train/ here "
                         "has no real landmark annotations regardless (see module docstring)")
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
        print(f"[done] train/: {n_train_images} images on disk (expect 12880; a handful of genuinely zero-face "
              f"images are dropped from label.txt, see _write_label_txt)")
    print(f"[done] eval tool: {'ready' if _eval_tool_ready() else 'NOT ready -- see errors above'}")


if __name__ == "__main__":
    main()
