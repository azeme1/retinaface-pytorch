"""Shared static-export wrapper + checkpoint loader used by every export
backend in this directory (export_coreml.py, export_onnx.py -- and, once
exported to ONNX, export_tflite.py/export_tfjs.py which convert THAT ONNX
rather than touching PyTorch again). Pulled out of export_coreml.py so all
formats trace the exact same graph and stay in sync automatically instead
of drifting copies -- see RetinaStaticExportWrapper's docstring for why the
graph looks the way it does.
"""

import json
import os
import sys
import zipfile
from pathlib import Path

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download

_RETINA_DIR = Path(__file__).resolve().parents[1]
_REPO_ROOT = Path("/workspace/home_0/work/llwll")

# This module is the INFERENCE side of the export pipeline and must have NO
# dependency on this project's private training/QAT package -- that package
# isn't even part of this repo. Every function here only ever sees a plain
# RetinaFace + a {table, indices} dict, never a raw training checkpoint.
if str(_RETINA_DIR) not in sys.path:
    sys.path.append(str(_RETINA_DIR))

from layers import PriorBox  # noqa: E402
from models import RetinaFace  # noqa: E402
from utils.box_utils import decode  # noqa: E402


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

    def forward(self, x: torch.Tensor):
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
        boxes = decode(loc, priors, self.variance) * self.bbox_scale
        landmarks = _decode_landmarks_2d(landmarks, priors, self.variance) * self.landmark_scale

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

    model = RetinaFace(cfg=cfg)
    state_dict = torch.load(plain_checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)  # strict=True: confirms this really is a plain checkpoint
    model.eval()
    cluster_info = torch.load(clusters_path, map_location="cpu", weights_only=False) if clusters_path else {}
    return model, cluster_info


def _hf_token(explicit: str | None = None) -> str | None:
    """Resolves a Hugging Face token: --hf-token if given, else the same
    .env_json this whole session's HF pushes have used, else HF_TOKEN env
    var, else None (fine for a public repo)."""
    if explicit:
        return explicit
    env_json = _REPO_ROOT / ".env_json"
    if env_json.exists():
        try:
            return json.loads(env_json.read_text()).get("HF_TOKEN")
        except Exception:
            pass
    return os.environ.get("HF_TOKEN")


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
