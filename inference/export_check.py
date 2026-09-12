"""Checks an ONNX/CoreML/TFLite export from export_onnx.py/export_coreml.py/
export_tflite.py against the PyTorch model it came from -- both a fast
numeric-parity spot check (raw box/score/landmark tensors on a handful of
images) and the real WIDER FACE AP (easy/medium/hard), computed
independently for the PyTorch wrapper and the converted format over the SAME
fixed-size preprocessing (this project's own resize/decode machinery in
widerface_eval.py assumes RetinaFace's own variable-size aspect-preserving
resize, which RetinaStaticExportWrapper deliberately does NOT use -- see its
docstring -- so AP here is only meaningful as an apples-to-apples PyTorch-vs-
converted-format comparison at THIS export's fixed image_size, not as a
number to compare against results/<network>/full_eval_parallel/*.json).

One script covering all three formats (replaces the former separate
export_onnx_check.py / export_coreml_check.py, which were ~90% identical --
only the "how do I load this artifact and run one image through it" part
ever differed): pass --format {onnx,coreml,tflite}.

Running the CoreML side needs coremltools' native runtime
(libcoremlpython), which only exists on macOS -- on any other platform this
script still builds/exports the PyTorch side and reports that clearly
rather than crashing on the missing runtime. ONNX Runtime and the TFLite
interpreter have no such platform restriction.

Usage:
    python inference/export_onnx.py --network mobilenetv1 \\
        --checkpoint pytorch_export/mobilenetv1_c7.zip --image-size 640 --out-dir onnx_export
    python inference/export_check.py --format onnx --network mobilenetv1 \\
        --checkpoint pytorch_export/mobilenetv1_c7.zip \\
        --artifact onnx_export/mobilenetv1_c7.zip --image-size 640 --n-images 200 \\
        --onnx-provider CUDA

    # CoreML -- this is the form to run on a Mac, where .predict() actually works.
    # --compute-units is required (see its --help): ALL, CoreML's own default,
    # has confirmed GPU/ANE-only numerical bugs on this graph -- CPU_ONLY is
    # the one confirmed correct so far.
    python inference/export_check.py --format coreml --network resnet18 \\
        --hf-repo azemel/retinaface-xs --hf-level c2 --image-size 640 --n-images 60 \\
        --compute-units CPU_ONLY

    # TFLite:
    python inference/export_check.py --format tflite --network mobilenetv1 \\
        --hf-repo azemel/retinaface-xs --hf-level c7 --image-size 640 --n-images 200

    # or download either artifact from an arbitrary URL directly (e.g. a
    # private repo's own resolve/main/... link, needs --hf-token or $HF_TOKEN):
    python inference/export_check.py --format onnx --network mobilenetv1 --image-size 640 \\
        --checkpoint-url https://huggingface.co/azemel/retinaface-xs/resolve/main/results/mobilenetv1/pytorch/mobilenetv1_c12.zip \\
        --artifact-url https://huggingface.co/azemel/retinaface-xs/resolve/main/results/mobilenetv1/onnx/mobilenetv1_c12.zip
"""
from __future__ import annotations

import argparse
import os
import random
import shutil
import sys
import tempfile
import threading
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import tqdm
import cv2
import numpy as np
import torch

_RETINA_DIR = Path(__file__).resolve().parents[1]
if str(_RETINA_DIR) not in sys.path:
    sys.path.append(str(_RETINA_DIR))

from config import get_config  # noqa: E402
import widerface_eval as we  # noqa: E402

sys.path.append(str(Path(__file__).resolve().parent))
from export_common import (  # noqa: E402
    RetinaStaticExportWrapper, load_plain_with_clusters, load_plain_with_clusters_from_hf,
    download_hf_artifact, replace_leaky_relu, download_from_url, select_device,
    postprocess, letterbox_resize, unletterbox,
)

DATASET_FOLDER = str(_RETINA_DIR / "data/widerface/val/images/")
VAL_LIST = str(_RETINA_DIR / "data/widerface/val/wider_val.txt")
GT_DIR = str(_RETINA_DIR / "widerface_evaluation/ground_truth")

# The generic filename each format's zip is expected to contain -- matches
# export_onnx.py/export_coreml.py/export_tflite.py's own packaging
# convention (never the network/cluster name, so nothing about which
# checkpoint produced it leaks from the archive's own contents).
ARTIFACT_NAME = {"onnx": "model.onnx", "coreml": "model.mlpackage", "tflite": "model.tflite"}
# Whether calling this format's own inference entry point concurrently from
# multiple threads on ONE loaded instance is safe. ONNX Runtime sessions are
# documented thread-safe for .run(). CoreML's MLModel.predict() is NOT --
# confirmed directly: without a lock, a full-val run hangs indefinitely at 0%
# (deadlocks inside the native runtime, not a Python exception). TFLite's
# Interpreter is documented as NOT safe for concurrent invoke() either, same
# treatment. Image decode + the PyTorch forward pass still run concurrently
# across threads regardless -- only the native call itself is serialized.
NEEDS_LOCK = {"onnx": False, "coreml": True, "tflite": True}


def preprocess(img_bgr: np.ndarray, image_size: int, fmt: str,
               input_color_order: str = "bgr") -> tuple[np.ndarray, np.ndarray, float]:
    """Aspect-preserving letterbox into the fixed image_size x image_size
    canvas (see export_common.letterbox_resize) -- confirmed empirically
    that a naive non-aspect-preserving stretch here collapses real AP
    (87.33% -> 68.13% on mobilenetv1 c7, Hard nearly halved), so this is not
    a cosmetic choice. Returns (pt_input, backend_input, scale) -- scale is
    needed by unletterbox() to map detections back to this image's native
    resolution. pt_input is always 1,3,H,W float32 CHW (what the PyTorch
    wrapper expects); backend_input is format-specific:
      - onnx:   identical to pt_input (ONNX Runtime takes the same NCHW tensor)
      - coreml: H,W,3 uint8 RGB -- CoreML's native image input is always
                declared RGB regardless of input_color_order (see export_coreml.py)
      - tflite: 1,H,W,3 float32 NHWC BGR (onnx2tf's conversion lands the
                model on NHWC, not NCHW -- see export_tflite.py's docstring)
    """
    canvas_bgr, scale = letterbox_resize(img_bgr, image_size)

    if fmt == "coreml":
        canvas_rgb = cv2.cvtColor(canvas_bgr, cv2.COLOR_BGR2RGB)
        ordered = canvas_rgb if input_color_order == "rgb" else canvas_bgr
        pt_input = np.float32(ordered).transpose(2, 0, 1)[None]
        backend_input = canvas_rgb.astype(np.uint8)
    elif fmt == "tflite":
        pt_input = np.float32(canvas_bgr).transpose(2, 0, 1)[None]
        backend_input = canvas_bgr[None].astype(np.float32)
    else:  # onnx
        pt_input = np.float32(canvas_bgr).transpose(2, 0, 1)[None]
        backend_input = pt_input

    return pt_input, backend_input, scale


def load_backend(fmt: str, artifact_path: str, image_size: int, compute_units: str | None = None,
                  onnx_provider: str | None = None):
    """Loads the converted-format artifact and returns (state, can_run,
    source_label) -- state is whatever run_backend needs, can_run is False
    only for CoreML on a non-macOS platform (every other case either runs
    or raises immediately at load time)."""
    if fmt == "onnx":
        import onnxruntime as ort
        # Unlike CoreML's compute-units bug, no correctness issue has been
        # found forcing one ONNX Runtime provider over another -- CUDA has
        # matched PyTorch closely on every checkpoint tested. --onnx-provider
        # is still required (not auto-selected), for the same reason as
        # --compute-units: explicit is safer than silently picking whatever
        # happens to be installed, e.g. so a run on a shared GPU box doesn't
        # accidentally contend with something else training on that GPU.
        provider_map = {"CPU": ["CPUExecutionProvider"], "CUDA": ["CUDAExecutionProvider"]}
        assert onnx_provider in provider_map, (
            f"--onnx-provider is required for --format onnx, one of {sorted(provider_map)}"
        )
        session = ort.InferenceSession(artifact_path, providers=provider_map[onnx_provider])
        print(f"ONNX Runtime using: {session.get_providers()[0]}")
        return {"session": session, "input_name": session.get_inputs()[0].name}, True

    if fmt == "coreml":
        import coremltools as ct
        import PIL.Image
        # CoreML's own default (ALL, meaning "let CoreML pick ANE/GPU/CPU
        # per-op") is NOT reliable for this graph -- confirmed two distinct
        # numerical bugs on the GPU/ANE path: an exp() overflow in box
        # decode (fixed separately, see utils/box_utils.py's _MAX_EXP_INPUT
        # clamp) and, even after that fix, the classification head still
        # comes back near-uniform ~0.99 confidence on many checkpoints under
        # ALL (vs. PyTorch's normal near-0-except-real-detections pattern),
        # flooding NMS with false positives and collapsing AP to ~0% despite
        # otherwise-finite box coordinates. CPU_ONLY has been confirmed
        # correct (matches PyTorch closely) in every case tested so far --
        # that's why compute_units is a REQUIRED argument here, not a
        # silently-trusted default: pass it explicitly so nobody reports a
        # "CoreML is broken" number that's actually just this ALL-path bug.
        compute_units_map = {
            "CPU_ONLY": ct.ComputeUnit.CPU_ONLY,
            "CPU_AND_GPU": ct.ComputeUnit.CPU_AND_GPU,
            "CPU_AND_NE": ct.ComputeUnit.CPU_AND_NE,
            "ALL": ct.ComputeUnit.ALL,
        }
        assert compute_units in compute_units_map, (
            f"--compute-units is required for --format coreml, one of {sorted(compute_units_map)}"
        )
        mlmodel = ct.models.MLModel(artifact_path, compute_units=compute_units_map[compute_units])
        try:
            # Loading an .mlpackage (parsing its protobuf) succeeds on any
            # platform -- it's .predict() specifically that needs the
            # native macOS CoreML runtime (libcoremlpython). Probe that
            # here, once, rather than discovering it independently inside
            # every image in the thread pool below.
            dummy = np.zeros((image_size, image_size, 3), dtype=np.uint8)
            mlmodel.predict({"input": PIL.Image.fromarray(dummy, mode="RGB")})
            return {"mlmodel": mlmodel}, True
        except Exception as e:  # noqa: BLE001 -- reported, not fatal
            print(f"[warning] can't run this .mlpackage on this platform ({type(e).__name__}: {e}) -- "
                  f"coremltools' .predict() needs the native macOS CoreML runtime (libcoremlpython), "
                  f"not available here. Reporting PyTorch-only.")
            return {"mlmodel": mlmodel}, False

    if fmt == "tflite":
        import tensorflow as tf
        interp = tf.lite.Interpreter(model_path=artifact_path)
        interp.allocate_tensors()
        return {
            "interp": interp,
            "input_index": interp.get_input_details()[0]["index"],
            "output_details": interp.get_output_details(),
        }, True

    raise ValueError(fmt)


def run_backend(fmt: str, state: dict, backend_input: np.ndarray):
    if fmt == "onnx":
        return state["session"].run(None, {state["input_name"]: backend_input})

    if fmt == "coreml":
        import PIL.Image
        img = PIL.Image.fromarray(backend_input, mode="RGB")
        out = state["mlmodel"].predict({"input": img})
        return out["boxes"], out["scores"], out["landmarks"]

    if fmt == "tflite":
        interp = state["interp"]
        interp.set_tensor(state["input_index"], backend_input)
        interp.invoke()
        # TFLite's output tensor NAMES don't reliably line up with which is
        # which (confirmed: an alphabetical-vs-declared-order mismatch
        # inside onnx2tf's own name-copying) -- identify by last-dimension
        # size instead (4=boxes, 1=scores, 10=landmarks), same as every
        # examples/*_inference.py script for this format.
        outs = {d["shape"][-1]: interp.get_tensor(d["index"]) for d in state["output_details"]}
        return outs[4], outs[1], outs[10]

    raise ValueError(fmt)


def print_ap_report(aps: dict) -> None:
    """Traditional WIDER FACE benchmark reporting: Easy/Medium/Hard/Average
    AP as percentages (the convention used across WIDER FACE leaderboards
    and papers), not raw 0-1 fractions."""
    print(f"  Easy:    {aps['easy'] * 100:6.2f}%")
    print(f"  Medium:  {aps['medium'] * 100:6.2f}%")
    print(f"  Hard:    {aps['hard'] * 100:6.2f}%")
    print(f"  Average: {we.mean_ap(aps) * 100:6.2f}%")


def print_comparison_table(network: str, format_name: str, pt_aps: dict, other_aps: dict) -> None:
    """Side-by-side PyTorch-vs-converted-format table -- the actual point
    of this whole script: not just two separate AP reports, but how much
    (if any) the conversion cost."""
    print(f"\n=== {network}: PyTorch vs {format_name} ===")
    print(f"{'':<10}{'PyTorch':>10}{format_name:>10}{'Diff':>10}")
    for label, key in [("Easy", "easy"), ("Medium", "medium"), ("Hard", "hard")]:
        pt_v, o_v = pt_aps[key] * 100, other_aps[key] * 100
        print(f"{label:<10}{pt_v:>9.2f}%{o_v:>9.2f}%{o_v - pt_v:>+9.2f}%")
    pt_avg, o_avg = we.mean_ap(pt_aps) * 100, we.mean_ap(other_aps) * 100
    print(f"{'Average':<10}{pt_avg:>9.2f}%{o_avg:>9.2f}%{o_avg - pt_avg:>+9.2f}%")


def write_prediction(save_folder: Path, img_name: str, boxes, scores):
    save_name = save_folder / (img_name[:-4] + ".txt")
    save_name.parent.mkdir(parents=True, exist_ok=True)
    with open(save_name, "w") as fd:
        fd.write(save_name.name[:-4] + "\n")
        fd.write(f"{len(boxes)}\n")
        for box, score in zip(boxes, scores.reshape(-1)):
            x, y = int(box[0]), int(box[1])
            w, h = int(box[2]) - x, int(box[3]) - y
            fd.write(f"{x} {y} {w} {h} {float(score)}\n")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--format", required=True, choices=["onnx", "coreml", "tflite"])
    p.add_argument("--network", required=True)
    p.add_argument("--checkpoint", default=None, help="local export_pytorch.py output -- combined .zip or "
                                                        "bare .pth -- mutually exclusive with "
                                                        "--checkpoint-url/--hf-repo")
    p.add_argument("--checkpoint-url", default=None, help="download --checkpoint from this URL instead of a "
                                                            "local file, e.g. a direct link into a Hugging Face "
                                                            "repo's resolve/main/... path")
    p.add_argument("--artifact", default=None, help="local export_onnx.py/export_coreml.py/export_tflite.py "
                                                      "output for --format (.zip, or the raw .onnx/.mlpackage/"
                                                      ".tflite) -- mutually exclusive with --artifact-url/--hf-repo")
    p.add_argument("--artifact-url", default=None, help="download --artifact from this URL instead of a local file")
    p.add_argument("--hf-repo", default=None, help="Hugging Face repo id (e.g. azemel/retinaface-xs) to "
                                                     "pull BOTH the pytorch source and the --format artifact "
                                                     "from instead of local files -- requires --hf-level")
    p.add_argument("--hf-level", default=None, help="e.g. 'c7' or 'float32' (see "
                                                      "export_pytorch_batch_hf.py's tag convention) -- selects "
                                                      "results/<network>/{pytorch,<format>}/<network>_<level>.zip")
    p.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"),
                    help="bearer token for a private repo -- defaults to the HF_TOKEN system variable, used for "
                         "--hf-repo as well as --checkpoint-url/--artifact-url")
    p.add_argument("--input-color-order", default="bgr", choices=["bgr", "rgb"],
                    help="--format coreml only -- must match what export_coreml.py used to build this "
                         ".mlpackage. Ignored for onnx/tflite.")
    p.add_argument("--compute-units", default=None,
                    choices=["CPU_ONLY", "CPU_AND_GPU", "CPU_AND_NE", "ALL"],
                    help="--format coreml only, REQUIRED (no default) -- which compute backend CoreML runs "
                         "on. CoreML's own default, ALL (let it pick ANE/GPU/CPU per-op), is NOT reliable "
                         "for this graph: confirmed a GPU/ANE-only exp() overflow in box decode (separately "
                         "fixed) and, even after that fix, near-uniform ~0.99 confidence on many checkpoints "
                         "under ALL that collapses AP to ~0%% despite otherwise-sane box coordinates. "
                         "CPU_ONLY has been confirmed correct (matches PyTorch closely) on every checkpoint "
                         "tested -- use that unless you're specifically investigating the GPU/ANE numerics "
                         "bug itself. Required (not defaulted) so a broken ALL-path run is never silently "
                         "mistaken for a real CoreML export problem. Ignored for onnx/tflite.")
    p.add_argument("--onnx-provider", default=None, choices=["CPU", "CUDA"],
                    help="--format onnx only, REQUIRED (no default) -- which ONNX Runtime execution "
                         "provider to use. No correctness bug forces this the way --compute-units is "
                         "forced for coreml, but it's still explicit rather than auto-selected: e.g. so a "
                         "run doesn't silently grab CUDA and contend with something else training on a "
                         "shared GPU box. Also forces the PyTorch reference side onto the matching device "
                         "(select_device(force=...)) so this is a genuine same-device comparison, not "
                         "ONNX-on-CPU vs a GPU-computed reference. Ignored for coreml/tflite.")
    p.add_argument("--image-size", type=int, default=640)
    p.add_argument("--n-images", type=int, default=None,
                    help="WIDER FACE val images to sample -- default is the FULL val set (3226 images), the "
                         "only way to get an AP comparable across runs; pass a smaller number for a quick, "
                         "noisier spot check instead")
    args = p.parse_args()
    fmt = args.format
    assert fmt != "coreml" or args.compute_units, (
        "--compute-units is required for --format coreml (e.g. --compute-units CPU_ONLY) -- "
        "see its --help for why ALL isn't a safe default here"
    )
    assert fmt != "onnx" or args.onnx_provider, (
        "--onnx-provider is required for --format onnx (e.g. --onnx-provider CPU) -- see its --help"
    )

    cfg = dict(get_config(args.network))

    def _resolve(local: str | None, url: str | None, label: str) -> str | None:
        assert not (local and url), f"--{label} and --{label}-url are mutually exclusive"
        return str(download_from_url(url, token=args.hf_token)) if url else local

    if args.hf_repo:
        assert args.hf_level, "--hf-repo needs --hf-level"
        assert not args.checkpoint and not args.checkpoint_url and not args.artifact and not args.artifact_url, (
            "--hf-repo is mutually exclusive with --checkpoint/--checkpoint-url/--artifact/--artifact-url"
        )
        model, _ = load_plain_with_clusters_from_hf(cfg, args.hf_repo, args.network, args.hf_level,
                                                     token=args.hf_token)
        artifact_arg = str(download_hf_artifact(args.hf_repo, args.network, fmt, args.hf_level, token=args.hf_token))
    else:
        checkpoint_arg = _resolve(args.checkpoint, args.checkpoint_url, "checkpoint")
        artifact_arg = _resolve(args.artifact, args.artifact_url, "artifact")
        assert checkpoint_arg and artifact_arg, (
            "need --checkpoint/--checkpoint-url + --artifact/--artifact-url (or --hf-repo + --hf-level instead)"
        )
        model, _ = load_plain_with_clusters(cfg, checkpoint_arg)

    wrapper_kwargs = {}
    if fmt == "coreml":
        # CoreML export always uses fp16 priors + RGB internal-permute --
        # the PyTorch reference needs to match, or numeric parity would be
        # comparing two different graphs, not the same one on two backends.
        wrapper_kwargs = {"priors_dtype": torch.float16, "input_color_order": args.input_color_order}
    wrapper = RetinaStaticExportWrapper(model, cfg, image_size=(args.image_size, args.image_size),
                                        **wrapper_kwargs).eval()
    if fmt == "coreml":
        replace_leaky_relu(wrapper)

    force_device = {"CPU": "cpu", "CUDA": "cuda"}.get(args.onnx_provider) if fmt == "onnx" else None
    device = select_device(force=force_device)
    print(f"PyTorch reference running on: {device}")
    wrapper = wrapper.to(device)

    tmp_ctx = None
    artifact_source_label = artifact_arg
    artifact_name = ARTIFACT_NAME[fmt]
    if artifact_arg.endswith(".zip"):
        tmp_ctx = tempfile.TemporaryDirectory()
        with zipfile.ZipFile(artifact_arg) as zf:
            if artifact_name in zf.namelist():
                zf.extract(artifact_name, tmp_ctx.name)
            else:
                zf.extractall(tmp_ctx.name)  # coreml's .mlpackage is a directory tree, not one member
        artifact_path = str(Path(tmp_ctx.name) / artifact_name)
    else:
        artifact_path = artifact_arg

    backend_state, can_run = load_backend(fmt, artifact_path, args.image_size,
                                          compute_units=args.compute_units, onnx_provider=args.onnx_provider)

    with open(VAL_LIST) as f:
        all_images = [n.lstrip("/") for n in f.read().split()]
    sample = all_images if args.n_images is None else random.Random(0).sample(all_images, args.n_images)

    pt_pred_dir = _RETINA_DIR / "results" / args.network / f"{fmt}_check_pred_pytorch"
    other_pred_dir = _RETINA_DIR / "results" / args.network / f"{fmt}_check_pred_{fmt}"
    for d in (pt_pred_dir, other_pred_dir):
        if d.exists():
            shutil.rmtree(d)
        # run_widerface_evaluation looks up every image in every WIDER FACE
        # event from the official fixed GT list -- an event with none of the
        # sampled images written yet would KeyError. Empty (0-detection)
        # placeholders for every val image first, only the sampled subset
        # gets overwritten with real predictions below.
        for img_name in all_images:
            write_prediction(d, img_name, np.zeros((0, 4)), np.zeros((0,)))

    # MPS has a history of thread-safety issues under concurrent inference
    # from multiple Python threads, same class of problem as the
    # CoreML/TFLite native-call locks below -- guarded defensively rather
    # than waiting to reproduce an MPS-specific hang. CUDA's concurrent-
    # inference-from-multiple-threads path is well-established, left unlocked.
    pytorch_lock = threading.Lock() if device.type == "mps" else None
    backend_lock = threading.Lock() if NEEDS_LOCK[fmt] else None

    def run_pytorch(pt_input: np.ndarray):
        x = torch.from_numpy(pt_input).to(device)
        with torch.no_grad():
            if pytorch_lock is not None:
                with pytorch_lock:
                    boxes, scores, landmarks = wrapper(x)
            else:
                boxes, scores, landmarks = wrapper(x)
        return boxes.cpu().numpy(), scores.cpu().numpy(), landmarks.cpu().numpy()

    def run_other(backend_input: np.ndarray):
        if backend_lock is not None:
            with backend_lock:
                return run_backend(fmt, backend_state, backend_input)
        return run_backend(fmt, backend_state, backend_input)

    diffs = {"boxes": [], "scores": [], "landmarks": []}

    def process(name: str):
        img_bgr = cv2.imread(str(Path(DATASET_FOLDER) / name), cv2.IMREAD_COLOR)
        if img_bgr is None:
            return None
        pt_input, backend_input, scale = preprocess(img_bgr, args.image_size, fmt, args.input_color_order)

        pt_boxes, pt_scores, pt_landm = run_pytorch(pt_input)
        d_boxes, d_scores, d_landm = postprocess(pt_boxes, pt_scores, pt_landm)
        d_boxes, d_landm = unletterbox(d_boxes, d_landm, scale)
        write_prediction(pt_pred_dir, name, d_boxes, d_scores)

        if not can_run:
            return None

        o_boxes_raw, o_scores_raw, o_landm_raw = run_other(backend_input)
        o_boxes, o_scores, o_landm = postprocess(o_boxes_raw, o_scores_raw, o_landm_raw)
        o_boxes, o_landm = unletterbox(o_boxes, o_landm, scale)
        write_prediction(other_pred_dir, name, o_boxes, o_scores)

        return {
            "boxes": float(np.abs(pt_boxes - o_boxes_raw).max()),
            "scores": float(np.abs(pt_scores - o_scores_raw).max()),
            "landmarks": float(np.abs(pt_landm - o_landm_raw).max()),
        }

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(process, name): name for name in sample}
        for fut in tqdm.tqdm(as_completed(futures), total=len(futures), desc=f"PyTorch vs {fmt.upper()}"):
            parity = fut.result()
            if parity is None:
                continue
            for k in diffs:
                diffs[k].append(parity[k])

    if len(sample) < len(all_images):
        print(f"\nNote: only {len(sample)}/{len(all_images)} val images were actually run -- every other image "
              f"counts as 0 recall (same downward bias as this repo's own training-time quick-probe), so the "
              f"absolute AP below is NOT comparable to results/<network>/full_eval_parallel/*.json. The bias is "
              f"identical on both sides, sampled from the same fixed seed, so the PyTorch-vs-{fmt.upper()} "
              f"COMPARISON is still meaningful -- omit --n-images for the full val set instead.")

    print(f"\n=== {args.network}: PyTorch (fixed-size wrapper) real WIDER FACE AP, n={len(sample)} images ===")
    pt_aps = we.run_widerface_evaluation(str(pt_pred_dir), GT_DIR)
    print_ap_report(pt_aps)

    if can_run:
        print(f"\n=== {args.network}: {fmt.upper()} ({artifact_source_label}) real WIDER FACE AP, n={len(sample)} images ===")
        other_aps = we.run_widerface_evaluation(str(other_pred_dir), GT_DIR)
        print_ap_report(other_aps)

        print_comparison_table(args.network, fmt.upper(), pt_aps, other_aps)

        print("\n=== numeric parity (raw tensors, before NMS/thresholding) ===")
        for k, vals in diffs.items():
            print(f"{k}: max={max(vals):.3e} mean={sum(vals)/len(vals):.3e}")
    else:
        print(f"\n({fmt.upper()}-side AP and parity skipped -- see warning above)")

    if tmp_ctx is not None:
        tmp_ctx.cleanup()


if __name__ == "__main__":
    main()
