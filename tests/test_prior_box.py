"""Parity tests: PriorBoxVectorized must produce identical anchors to the
original PriorBox for every configuration it's used as a drop-in
replacement for -- same shape, same values, same dtype, same iteration
order.
"""
import pytest
import torch

from config import cfg_mnet, cfg_mnet_025, cfg_mnet_050, cfg_mnet_v2, cfg_re18, cfg_re34, cfg_re50
from layers import PriorBox
from layers.functions.prior_box import PriorBoxVectorized

ALL_CFGS = [cfg_mnet, cfg_mnet_025, cfg_mnet_050, cfg_mnet_v2, cfg_re18, cfg_re34, cfg_re50]

IMAGE_SIZES = [
    (640, 640),   # standard training crop, square, evenly divisible by every step
    (320, 320),   # smaller square
    (1600, 1200), # WIDER FACE eval-style upscale, rectangular
    (641, 641),   # not evenly divisible by any step -- exercises the ceil() edge cell
    (100, 50),    # small and rectangular
]


def _make_cfg(base_cfg: dict, clip: bool) -> dict:
    return {**base_cfg, "clip": clip}


@pytest.mark.parametrize("base_cfg", ALL_CFGS, ids=lambda c: c["name"])
@pytest.mark.parametrize("image_size", IMAGE_SIZES, ids=lambda s: f"{s[0]}x{s[1]}")
@pytest.mark.parametrize("clip", [False, True])
def test_vectorized_matches_original(base_cfg, image_size, clip):
    cfg = _make_cfg(base_cfg, clip)

    original = PriorBox(cfg, image_size=image_size).generate_anchors()
    vectorized = PriorBoxVectorized(cfg, image_size=image_size).generate_anchors()

    assert original.shape == vectorized.shape
    assert original.dtype == vectorized.dtype == torch.float32
    torch.testing.assert_close(original, vectorized, rtol=0, atol=1e-6)


def test_vectorized_matches_original_uneven_min_sizes():
    """min_sizes lists of different lengths per feature-map level -- makes
    sure the per-level repeat/tile counts aren't accidentally hardcoded to
    a uniform anchors-per-cell count."""
    cfg = {
        "clip": False,
        "steps": [8, 16, 32],
        "min_sizes": [[16], [64, 96, 128], [256, 384, 512, 640]],
    }
    image_size = (640, 480)

    original = PriorBox(cfg, image_size=image_size).generate_anchors()
    vectorized = PriorBoxVectorized(cfg, image_size=image_size).generate_anchors()

    assert original.shape == vectorized.shape
    torch.testing.assert_close(original, vectorized, rtol=0, atol=1e-6)


def test_vectorized_matches_original_single_level():
    cfg = {"clip": True, "steps": [16], "min_sizes": [[32, 64]]}
    image_size = (256, 256)

    original = PriorBox(cfg, image_size=image_size).generate_anchors()
    vectorized = PriorBoxVectorized(cfg, image_size=image_size).generate_anchors()

    torch.testing.assert_close(original, vectorized, rtol=0, atol=1e-6)
