import math
import numpy as np
from itertools import product

import torch
from typing import Tuple


class PriorBox:
    def __init__(self, cfg: dict, image_size: Tuple[int, int]) -> None:
        super().__init__()
        self.image_size = image_size
        self.clip = cfg['clip']
        self.steps = cfg['steps']
        self.min_sizes = cfg['min_sizes']
        self.feature_maps = [[
            math.ceil(self.image_size[0]/step), math.ceil(self.image_size[1]/step)] for step in self.steps
        ]

    def generate_anchors(self) -> torch.Tensor:
        """Generate anchor boxes based on configuration and image size"""
        anchors = []
        for k, (map_height, map_width) in enumerate(self.feature_maps):
            step = self.steps[k]
            for i, j in product(range(map_height), range(map_width)):
                for min_size in self.min_sizes[k]:
                    s_kx = min_size / self.image_size[1]
                    s_ky = min_size / self.image_size[0]

                    dense_cx = [x * step / self.image_size[1] for x in [j+0.5]]
                    dense_cy = [y * step / self.image_size[0] for y in [i+0.5]]
                    for cy, cx in product(dense_cy, dense_cx):
                        anchors += [cx, cy, s_kx, s_ky]

        # back to torch land
        output = torch.Tensor(anchors).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output


class PriorBoxVectorized:
    """Numpy-vectorized drop-in replacement for PriorBox.

    Produces numerically identical output to PriorBox (same (level, row,
    col, min_size) iteration order) but without PriorBox's per-anchor
    Python-level list-append. That loop was confirmed to be the actual
    bottleneck for WIDER FACE eval (not the model forward pass, despite
    looking GPU-bound at a glance): eval images resize to ~1600px on the
    short side (vs. training's fixed 640 crop), which produces roughly
    79,000 anchors for a single ~1600x1200 image -- ~255 MILLION Python-level
    loop iterations over a 3226-image full-val pass. That loop is pure
    Python (no numpy/C release of the GIL), so it stayed fully
    single-threaded no matter how many worker threads the surrounding eval
    code used -- sampling GPU utilization repeatedly (not a single snapshot)
    showed brief ~100% spikes (the actual forward pass, genuinely fast)
    followed by many seconds at ~0% (this loop, per image, unparallelizable
    by more threads). Use this class for high-anchor-count workloads (e.g.
    full-resolution eval); use PriorBox when parity with the original
    reference implementation matters more than speed.
    """

    def __init__(self, cfg: dict, image_size: Tuple[int, int]) -> None:
        super().__init__()
        self.image_size = image_size
        self.clip = cfg['clip']
        self.steps = cfg['steps']
        self.min_sizes = cfg['min_sizes']
        self.feature_maps = [[
            math.ceil(self.image_size[0]/step), math.ceil(self.image_size[1]/step)] for step in self.steps
        ]

    def generate_anchors(self) -> torch.Tensor:
        """Generate anchor boxes based on configuration and image size."""
        anchors = []
        img_h, img_w = self.image_size
        for k, (map_height, map_width) in enumerate(self.feature_maps):
            step = self.steps[k]
            min_sizes = self.min_sizes[k]
            n_ms = len(min_sizes)

            # cx depends only on column j, cy only on row i -- same values
            # the original's per-(i,j) single-element dense_cx/dense_cy
            # lists computed, just built once per axis instead of per cell.
            cx = (np.arange(map_width) + 0.5) * step / img_w
            cy = (np.arange(map_height) + 0.5) * step / img_h

            # meshgrid + reshape(-1) with default 'xy' indexing gives a
            # (map_height, map_width) grid where row i holds cy[i] and
            # column j holds cx[j]; C-order flatten iterates j fastest
            # within each i -- identical to the original's
            # product(range(map_height), range(map_width)) order (i outer,
            # j inner).
            cx_grid, cy_grid = np.meshgrid(cx, cy)
            cx_flat = cx_grid.reshape(-1)
            cy_flat = cy_grid.reshape(-1)

            # Original innermost loop was "for min_size in min_sizes[k]"
            # for each (i,j) -- repeat each (i,j) center n_ms times
            # (consecutive) and tile the min_size sequence once per (i,j)
            # to match that exact per-cell ordering.
            cx_rep = np.repeat(cx_flat, n_ms)
            cy_rep = np.repeat(cy_flat, n_ms)
            s_kx = np.tile(np.asarray(min_sizes, dtype=np.float64) / img_w, cx_flat.size)
            s_ky = np.tile(np.asarray(min_sizes, dtype=np.float64) / img_h, cx_flat.size)

            level_anchors = np.stack([cx_rep, cy_rep, s_kx, s_ky], axis=1)
            anchors.append(level_anchors)

        anchors = np.concatenate(anchors, axis=0) if anchors else np.zeros((0, 4))

        # back to torch land
        output = torch.from_numpy(anchors).float()
        if self.clip:
            output.clamp_(max=1, min=0)
        return output
