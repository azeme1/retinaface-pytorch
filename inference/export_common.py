"""Shared static-export wrapper + checkpoint loader used by every export
backend in this directory (export_coreml.py, export_onnx.py -- and, once
exported to ONNX, export_tflite.py/export_tfjs.py which convert THAT ONNX
rather than touching PyTorch again). Pulled out of export_coreml.py so all
formats trace the exact same graph and stay in sync automatically instead
of drifting copies -- see RetinaStaticExportWrapper's docstring for why the
graph looks the way it does.
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.ops
from huggingface_hub import hf_hub_download


from layers import PriorBox  # noqa: E402
from models import RetinaFace  # noqa: E402
from utils.box_utils import decode  # noqa: E402


def select_device(force: str | None = None) -> torch.device:
    """Best available device for eager PyTorch inference: CUDA, then Apple
    Silicon MPS, else CPU -- used by the check scripts' "PyTorch reference"
    side (the exported CoreML/ONNX/etc. artifact runs on its own separate
    engine regardless of this). Pass force="cpu"/"cuda"/"mps" to skip
    auto-selection (e.g. export_check.py's --format onnx forces this to
    match --onnx-provider, so a --onnx-provider CPU run doesn't compare a
    CPU-side ONNX Runtime session against a GPU-computed "reference" --
    asserts the requested device is actually available rather than silently
    falling back)."""
    if force is not None:
        assert force in ("cpu", "cuda", "mps"), f"force must be cpu/cuda/mps, got {force!r}"
        if force == "cuda":
            assert torch.cuda.is_available(), "force='cuda' but CUDA is not available"
        if force == "mps":
            assert torch.backends.mps.is_available(), "force='mps' but MPS is not available"
        return torch.device(force)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class _SafeLeakyReLU(nn.Module):
    """Drop-in replacement for nn.LeakyReLU that avoids coremltools 9.0's
    mlprogram frontend bug converting aten::leaky_relu(_): it always infers
    the `alpha` (negative_slope) operand as int32 regardless of the actual
    float value, then rejects the op for not matching x's fp32 dtype
    (confirmed for both the in-place and out-of-place variants). relu/mul/
    sub/where all convert natively with no such issue, and this is exactly
    leaky_relu's own definition, so it's numerically identical (eval mode,
    no autograd involved)."""

    def __init__(self, negative_slope: float):
        super().__init__()
        self.negative_slope = negative_slope

    def forward(self, x):
        return torch.where(x >= 0, x, x * self.negative_slope)


def replace_leaky_relu(module: nn.Module) -> None:
    """In-place: swaps every nn.LeakyReLU in `module` for _SafeLeakyReLU.
    Call this right before tracing for CoreML (see export_coreml.py) -- the
    other export targets (ONNX/TFLite/TFJS) convert plain LeakyReLU fine."""
    for parent in module.modules():
        for name, child in list(parent.named_children()):
            if isinstance(child, nn.LeakyReLU):
                setattr(parent, name, _SafeLeakyReLU(child.negative_slope))


def _decode_landmarks_2d(predictions: torch.Tensor, priors: torch.Tensor, variance: list[float]) -> torch.Tensor:
    """Same arithmetic as utils.box_utils.decode_landmarks (confirmed
    bit-identical: every element is an independent multiply-add, so
    reshaping changes nothing about the floating-point result), rebuilt
    with only 2D slice/cat/broadcast ops instead of that function's
    view(N, 5, 2) + unsqueeze(1) 3D reshape. Needed because pnnx's ncnn
    backend (see export_ncnn.py) miscompiles the 3D version -- confirmed
    it emits a transposed reshape (0=10, 1=4200 instead of the reverse)
    feeding a BinaryOp, which segfaults ncnn's own broadcast implementation
    at inference time. ONNX/CoreML/TFLite/TFJS all handle the original
    fine, but there's no reason to keep two versions to maintain: this one
    exports cleanly to every backend, so it's used everywhere.
    """
    center = torch.cat([priors[:, 0:1], priors[:, 1:2]], dim=1).repeat(1, 5)  # (N,10): cx,cy,cx,cy,...
    scale = torch.cat([priors[:, 2:3], priors[:, 3:4]], dim=1).repeat(1, 5)   # (N,10): w,h,w,h,...
    return center + predictions * variance[0] * scale


class RetinaStaticExportWrapper(nn.Module):
    """Fuses PriorBox as a registered buffer + bakes box/landmark decode
    into forward() -- computes the prior grid ONCE for a fixed export image
    size and treats it as a weight, following
    https://github.com/azeme1/Pytorch_Retinaface_BBFree/blob/master/convert_to_onnx_original.py's
    RetinaStaticExportWrapper -- instead of recomputing it via
    PriorBox(...).generate_anchors() inside forward() every call. This
    keeps ONNX/CoreML/TFLite/TFJS consumers from having to reimplement
    anchor generation themselves: the exported graph takes a raw
    (mean-subtracted internally) image and returns absolute-coordinate
    boxes/scores/landmarks for EVERY prior directly.

    Confidence filtering and NMS stay OUTSIDE the graph -- same choice this
    project's external/face_detector/model.py already documents for
    RetinaFace's OTHER export path (decode-free there; here decode is fused
    but NMS still isn't): keeps the traced graph static-shaped and friendly
    to every one of these converters, since a data-dependent
    variable-length NMS output would fight all of them (ONNX's opset NMS op
    has spotty runtime support, CoreML's iOS-only, TFLite/TFJS need a
    custom op) -- every example under examples/ does confidence threshold +
    NMS itself, in plain numpy/JS, against this same fixed-length output.
    """

    def __init__(self, model: nn.Module, cfg: dict, image_size: tuple[int, int],
                 priors_dtype: torch.dtype = torch.float32, input_color_order: str = "bgr"):
        """input_color_order describes what channel order the INCOMING tensor
        is in, independent of how this checkpoint's weights were trained
        (always BGR in this repo, cv2's convention -- see rgb_mean below).
        "bgr": no-op, input already matches training. "rgb": a platform's
        native image pipeline (e.g. CoreML's ct.ImageType, which every
        export here declares as RGB regardless of any given checkpoint's own
        training color order -- see export_coreml.py) hands this model a
        different channel order than it was trained on -- permuted back to
        BGR right here, once, before anything else runs, so every other line
        of this class (and the wrapped model itself) never has to care."""
        super().__init__()
        if input_color_order not in ("bgr", "rgb"):
            raise ValueError(f"input_color_order must be 'bgr' or 'rgb', got {input_color_order!r}")
        self.model = model
        self.variance = cfg["variance"]
        self.input_color_order = input_color_order

        priors = PriorBox(cfg, image_size=image_size).generate_anchors()  # (num_priors, 4), normalized
        # Fused like a weight: computed ONCE here for this fixed export
        # size, stored as a genuine buffer (moves with .to()/.eval(),
        # included in state_dict, gets traced as a graph CONSTANT --
        # never recomputed at inference time). CoreML export passes
        # torch.float16 here to halve this buffer's footprint (its own
        # prior request); ONNX/TFLite/TFJS keep the float32 default since
        # those converters otherwise insert extra fp16<->fp32 cast ops
        # around a lone half-precision constant.
        self.register_buffer("priors", priors.to(priors_dtype))
        rgb_mean = torch.tensor([104.0, 117.0, 123.0]).view(1, 3, 1, 1)  # bgr order, matches train.py
        self.register_buffer("rgb_mean", rgb_mean)

        # Plain Python constants, not derived from x.shape inside forward():
        # this is a FIXED-size static export, so the height/width used to
        # rescale decoded (0-1 normalized) boxes/landmarks back to pixel
        # coordinates are already known here. Tracing `_, _, height, width
        # = x.shape` instead produces a dynamic aten::size graph node --
        # confirmed coremltools' MIL converter can't cast that traced value
        # back to a plain int, and onnx2tf's shape inference gets similarly
        # confused turning a dynamic Slice/Gather chain into TF ops -- even
        # though the actual value never varies for this wrapper.
        height, width = image_size
        bbox_scale = torch.tensor([width, height, width, height], dtype=torch.float32)
        self.register_buffer("bbox_scale", bbox_scale)
        landmark_scale = torch.tensor([width, height] * 5, dtype=torch.float32)
        self.register_buffer("landmark_scale", landmark_scale)

    def forward(self, x: torch.Tensor, output_size: tuple[int, int] | None = None):
        """output_size: (width, height) to rescale boxes/landmarks to,
        e.g. an original image's own native resolution -- OFF (None) by
        default, which is what every actual export traces with, so this
        adds no op to the exported ONNX/CoreML/etc. graph (those always
        return export-canvas-space coordinates; an exported graph can't
        know an arbitrary caller's native size at conversion time anyway).
        This is purely an eager-PyTorch convenience for callers that DO
        know their own target size up front (e.g. a check/comparison
        script running the un-exported wrapper directly against images of
        varying native resolution) -- pass it and get ready-to-use,
        already-scaled boxes back instead of manually rescaling
        export-canvas coordinates afterward (see examples/_postprocess.py's
        rescale_to_original for the equivalent standalone helper every
        exported-format consumer still needs, since THEY have no such
        option)."""
        # x: [1, 3, H, W], raw float pixel values (0-255) in
        # self.input_color_order channel order -- normalized here so the
        # exported model is fully self-contained (raw image in), same
        # convention as this project's face_detector/model.py.
        if self.input_color_order == "rgb":
            x = x[:, [2, 1, 0], :, :]  # -> bgr, matching training (see __init__)
        x = x - self.rgb_mean
        loc, conf, landmarks = self.model(x)
        # [0] instead of .squeeze(0): mathematically identical for this
        # guaranteed-batch-size-1 export, but pnnx's ncnn backend (see
        # export_ncnn.py) silently mis-lowers squeeze(0) on a batch axis
        # ("squeeze batch dim 0 is not supported yet!", then produces
        # wrong numbers rather than erroring) -- confirmed indexing avoids
        # it, and every other backend traces it identically either way.
        loc, conf, landmarks = loc[0], conf[0], landmarks[0]

        priors = self.priors.float()  # upcast needed when priors_dtype is float16 (CoreML export)
        bbox_scale, landmark_scale = self.bbox_scale, self.landmark_scale
        if output_size is not None:
            # self.bbox_scale is [canvas_w, canvas_h, canvas_w, canvas_h] --
            # rescaling relative to it (rather than re-deriving canvas size
            # separately) keeps this exact to whatever forward() already
            # decodes with, no risk of drifting out of sync with it.
            out_w, out_h = output_size
            canvas_w, canvas_h = float(bbox_scale[0]), float(bbox_scale[1])
            bbox_scale = bbox_scale * torch.tensor(
                [out_w / canvas_w, out_h / canvas_h, out_w / canvas_w, out_h / canvas_h])
            landmark_scale = landmark_scale * torch.tensor([out_w / canvas_w, out_h / canvas_h] * 5)
        boxes = decode(loc, priors, self.variance) * bbox_scale
        landmarks = _decode_landmarks_2d(landmarks, priors, self.variance) * landmark_scale

        scores = conf[:, 1:2]  # face-class confidence, kept 2D: [num_priors, 1]
        return boxes, scores, landmarks


def cluster_count(cluster_info: dict) -> int | None:
    """Number of clusters per layer, read off cluster_info's per-layer
    codebook table -- None for an empty dict (float32, nothing quantized).
    "table" is the current key name for that codebook; older staged
    artifacts (converted before that rename) still carry the previous key,
    accepted here too so they don't need to be reconverted."""
    if not cluster_info:
        return None
    entry = next(iter(cluster_info.values()))
    key = "table" if "table" in entry else "lut"
    return entry[key].shape[-1]


def _build_from_payload(cfg: dict, payload: dict) -> tuple[nn.Module, dict]:
    """Shared by every "combined zip" loader below: payload is the
    {"state_dict": ..., "clusters": ...} dict export_pytorch_batch_hf.py
    packs into one checkpoint.pth, regardless of whether it arrived via a
    local zip or a Hugging Face download."""
    # pretrain=False: the backbone's ImageNet-pretrained init is about to be
    # overwritten by load_state_dict below anyway, but building it still
    # tries to read a local weights/<backbone>.pretrained file first -- one
    # that isn't shipped for every backbone (confirmed missing for the
    # mobilenetv1_0.25/0.50 width variants), crashing export for those with
    # a FileNotFoundError before load_state_dict ever runs.
    cfg = dict(cfg, pretrain=False)
    model = RetinaFace(cfg=cfg)
    model.load_state_dict(payload["state_dict"])  # strict=True
    model.eval()
    return model, payload.get("clusters", {})


def load_plain_with_clusters_from_zip(cfg: dict, zip_path: str) -> tuple[nn.Module, dict]:
    """Loads directly from ONE combined zip (export_pytorch_batch_hf.py's own
    packing: a single generically-named checkpoint.pth inside holding
    {"state_dict": ..., "clusters": ...}) -- the caller doesn't need to
    pre-split it into two files first; this does whatever unzipping/loading
    is required from that one input. Use this (or load_plain_with_clusters,
    which now also accepts a zip directly) instead of asking for two
    separate --plain-checkpoint/--clusters paths."""
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        assert len(names) == 1, f"{zip_path} has more than one member: {names}"
        with zf.open(names[0]) as f:
            payload = torch.load(f, map_location="cpu", weights_only=False)
    return _build_from_payload(cfg, payload)


def load_plain_with_clusters(cfg: dict, plain_checkpoint: str, clusters_path: str | None = None) -> tuple[nn.Module, dict]:
    """The canonical entry point for every export format EXCEPT the plain
    PyTorch one itself: this repo's plain+clusters checkpoint (produced
    upstream, outside this repo, from a raw training checkpoint) is the
    single source every other backend (CoreML, ONNX, ...) should build
    from. This function, and everything downstream of it, only ever sees a
    stock RetinaFace plus a plain {table, indices} dict -- never a raw
    training checkpoint or any of this project's private training machinery.

    plain_checkpoint accepts EITHER form export_pytorch.py can produce:
    a bare .pth (state_dict only -- pair it with clusters_path if this
    level was quantized) or a single combined .zip (export_pytorch_batch_hf.py's
    packing, {"state_dict": ..., "clusters": ...} in one file -- clusters_path
    is ignored in that case, the zip already carries both).
    """
    if str(plain_checkpoint).endswith(".zip"):
        return load_plain_with_clusters_from_zip(cfg, plain_checkpoint)

    model = RetinaFace(cfg=dict(cfg, pretrain=False))  # see _build_from_payload's comment on pretrain=False
    state_dict = torch.load(plain_checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)  # strict=True: confirms this really is a plain checkpoint
    model.eval()
    cluster_info = torch.load(clusters_path, map_location="cpu", weights_only=False) if clusters_path else {}
    return model, cluster_info


def _hf_token(explicit: str | None = None) -> str | None:
    """Resolves a Hugging Face token: --hf-token if given, else the
    HF_TOKEN system variable, else None (fine for a public repo)."""
    return explicit or os.environ.get("HF_TOKEN")


def download_hf_artifact(repo_id: str, network: str, fmt: str, level: str, token: str | None = None) -> Path:
    """Downloads one exported artifact already published to the given HF
    repo, under the layout this session's own HF pushes use:
    results/<network>/<fmt>/<network>_<level>.zip -- fmt is "pytorch",
    "onnx", or "coreml"; level is e.g. "c7" or "float32" (a cluster count
    or "float32", never any hint of the training method that produced it).
    Returns the local cached file path (huggingface_hub's own cache, so a
    second call for the same file is instant, no re-download)."""
    path_in_repo = f"results/{network}/{fmt}/{network}_{level}.zip"
    return Path(hf_hub_download(repo_id=repo_id, repo_type="model", filename=path_in_repo,
                                 token=_hf_token(token)))


def load_plain_with_clusters_from_hf(cfg: dict, repo_id: str, network: str, level: str,
                                      token: str | None = None) -> tuple[nn.Module, dict]:
    """Same contract as load_plain_with_clusters (a plain RetinaFace + its
    cluster info, if any) but sourced from a Hugging Face repo instead of a
    local file -- downloads the combined zip and hands it to
    load_plain_with_clusters_from_zip."""
    zip_path = download_hf_artifact(repo_id, network, "pytorch", level, token=token)
    return load_plain_with_clusters_from_zip(cfg, str(zip_path))


def download_from_url(url: str, token: str | None = None) -> Path:
    """Downloads any artifact (a plain+clusters checkpoint zip/.pth, or a
    zipped .mlpackage) straight from a URL to a local temp file, preserving
    the URL's own suffix so callers' own `.endswith(".zip")` dispatch still
    works -- e.g. a direct link into a Hugging Face repo's resolve/main/...
    path (export_pytorch_batch_hf.py's/export_coreml.py's own naming
    convention), but works for any plain HTTP(S) URL. Sends the given (or
    HF_TOKEN system variable) token as a bearer token if set -- needed for
    a private HF repo (a plain unauthenticated GET against one 401s),
    harmless against a public URL that ignores it."""
    suffix = Path(url.split("?")[0]).suffix or ".zip"
    tmp_path = Path(tempfile.mkstemp(suffix=suffix)[1])
    print(f"downloading {url} ...")
    token = _hf_token(token)
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as resp, open(tmp_path, "wb") as out:
        shutil.copyfileobj(resp, out)
    print(f"downloaded -> {tmp_path} ({tmp_path.stat().st_size / 1e6:.2f} MB)")
    return tmp_path


def nms(dets, thresh: float) -> list[int]:
    """dets: [N,5] (x1,y1,x2,y2,score). torchvision.ops.nms's own CPU/CUDA
    kernel -- battle-tested, no hand-rolled IoU/division-by-zero edge cases
    to get wrong -- instead of a custom numpy re-implementation."""
    boxes = torch.as_tensor(dets[:, :4], dtype=torch.float32)
    scores = torch.as_tensor(dets[:, 4], dtype=torch.float32)
    return torchvision.ops.nms(boxes, scores, thresh).tolist()


def postprocess(boxes, scores, landmarks, conf_threshold: float = 0.5, nms_threshold: float = 0.4, top_k: int = 750):
    """boxes: [N,4], scores: [N,1] or [N], landmarks: [N,10] -- raw decoded
    output straight from any of this repo's exported models (or the
    RetinaStaticExportWrapper directly). Returns (boxes, scores, landmarks)
    for the detections that survive thresholding + NMS, highest score
    first. Same implementation as examples/_postprocess.py's own
    postprocess() -- duplicated here (not imported from there) so the
    inference/ scripts in this directory don't reach into a sibling
    directory for something this basic; examples/ stays self-contained
    for its own, separate audience (someone who only downloaded that one
    folder from the Hub)."""
    scores = scores.reshape(-1)
    keep = scores > conf_threshold
    boxes, scores, landmarks = boxes[keep], scores[keep], landmarks[keep]

    order = scores.argsort()[::-1]
    boxes, scores, landmarks = boxes[order], scores[order], landmarks[order]

    dets = np.hstack([boxes, scores[:, None]]).astype(np.float32, copy=False)
    keep_idx = nms(dets, nms_threshold)[:top_k]
    return boxes[keep_idx], scores[keep_idx], landmarks[keep_idx]


def rescale_to_original(boxes, landmarks, image_size: int, native_w: int, native_h: int):
    """Rescales boxes/landmarks from the fixed image_size x image_size
    export canvas (see RetinaStaticExportWrapper) back to one image's own
    native resolution, for a NON-aspect-preserving stretch-resize (x and y
    scaled independently). Superseded by letterbox_resize/unletterbox
    below for actual accuracy -- kept only for anything that deliberately
    still wants stretch behavior."""
    if boxes.shape[0] == 0:
        return boxes, landmarks
    sx, sy = native_w / image_size, native_h / image_size
    boxes = boxes.copy()
    boxes[:, [0, 2]] *= sx
    boxes[:, [1, 3]] *= sy
    landmarks = landmarks.copy()
    landmarks[:, 0::2] *= sx
    landmarks[:, 1::2] *= sy
    return boxes, landmarks


# BGR order, matches RetinaStaticExportWrapper's own rgb_mean buffer -- padding
# with exactly this value means the padded region becomes precisely 0 after
# the wrapper's internal mean-subtraction, contributing no spurious signal.
LETTERBOX_PAD_VALUE_BGR = (104, 117, 123)


def letterbox_resize(img_bgr, image_size: int, pad_value=LETTERBOX_PAD_VALUE_BGR):
    """Aspect-preserving resize into a fixed image_size x image_size canvas:
    scales the image down/up by ONE uniform factor (so shapes aren't
    distorted, unlike a plain stretch-resize) and places the result at the
    canvas's TOP-LEFT corner, padding only the bottom/right edges. Placing
    at (0, 0) instead of centering is deliberate -- it means unletterbox()
    below is a pure division by scale, no offset subtraction, so there is
    no separate padding math to get wrong when mapping detections back to
    the original image. Returns (canvas, scale) -- scale is what
    unletterbox() needs."""
    h, w = img_bgr.shape[:2]
    scale = min(image_size / w, image_size / h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((image_size, image_size, 3), pad_value, dtype=img_bgr.dtype)
    canvas[:new_h, :new_w] = resized
    return canvas, scale


def unletterbox(boxes, landmarks, scale: float):
    """Inverse of letterbox_resize's placement: boxes/landmarks come back
    in canvas pixel space, and since the resized image sits at the
    canvas's (0, 0) origin with no offset, mapping back to the original
    image is a single division by scale -- no padding offset involved."""
    if boxes.shape[0] == 0:
        return boxes, landmarks
    return boxes / scale, landmarks / scale
