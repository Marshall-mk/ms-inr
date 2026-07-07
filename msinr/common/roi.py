"""Region-of-interest (brain) cropping + sample masking for reconstruction.

Whole-head real data (not skull-stripped) makes the coordinate INR waste its
capacity on skull/scalp/neck, and the normalizer maps the whole FOV to [-1,1] so
the brain gets only a fraction of the frequency budget. Given a brain mask we:
  * crop the reconstruction grid to the brain bounding box (+margin) -> the
    normalizer now spans just the brain (2-3x effective resolution), and
  * drop training samples that fall outside the brain -> capacity focuses on it.
"""
from __future__ import annotations

import numpy as np

from .contracts import Volume, GridSpec
from .geometry import apply_affine
from scipy.ndimage import map_coordinates


def crop_grid_to_mask(grid: GridSpec, mask: Volume, margin_mm: float = 8.0) -> GridSpec:
    """Crop ``grid`` to the world bounding box of ``mask`` > 0 (+ margin), keeping
    the grid's spacing/orientation. ``mask`` may live on any grid."""
    m = mask.data > 0
    if not m.any():
        return grid
    # brain voxel coords -> world -> grid index coords
    idx = np.argwhere(m).astype(np.float64)
    world = apply_affine(mask.affine, idx)
    ginv = np.linalg.inv(grid.affine)
    gidx = apply_affine(ginv, world)
    spacing = np.linalg.norm(grid.affine[:3, :3], axis=0)
    pad = np.ceil(margin_mm / np.maximum(spacing, 1e-6)).astype(int)
    lo = np.floor(gidx.min(0)).astype(int) - pad
    hi = np.ceil(gidx.max(0)).astype(int) + pad
    lo = np.maximum(lo, 0)
    hi = np.minimum(hi, np.array(grid.shape) - 1)
    shape = tuple(int(h - l + 1) for l, h in zip(lo, hi))
    affine = grid.affine.copy()
    affine[:3, 3] = apply_affine(grid.affine, lo[None].astype(float))[0]  # new origin
    return GridSpec(shape=shape, affine=affine)


def maybe_crop_to_roi(grid: GridSpec, cfg: dict):
    """If cfg['roi_mask'] is set, load it and crop ``grid`` to the brain bbox.
    Returns (grid, roi_volume_or_None)."""
    if not cfg.get("roi_mask"):
        return grid, None
    from . import io as _io
    roi = _io.load_volume(cfg["roi_mask"], name="roi")
    return crop_grid_to_mask(grid, roi, margin_mm=cfg.get("roi_margin_mm", 8.0)), roi


def mask_at_world(mask: Volume, world_pts: np.ndarray) -> np.ndarray:
    """Boolean: is each world point inside the brain mask (nearest-neighbour)."""
    vox = apply_affine(np.linalg.inv(mask.affine), world_pts)
    vals = map_coordinates(mask.data.astype(np.float32), vox.T, order=0,
                           mode="constant", cval=0.0)
    return vals > 0.5
