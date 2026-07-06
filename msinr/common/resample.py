"""Resample a Volume onto an arbitrary target grid (affine + shape) by trilinear
interpolation in world space. Used to bring a method's reconstruction onto the GT
grid before computing metrics."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import map_coordinates

from .contracts import Volume, GridSpec
from .geometry import voxel_grid_world, apply_affine


def resample_to_grid(vol: Volume, grid: GridSpec, order: int = 1) -> Volume:
    world = voxel_grid_world(grid.shape, grid.affine)          # (P,3) target world
    vox = apply_affine(np.linalg.inv(vol.affine), world)       # source voxel coords
    data = map_coordinates(vol.data, vox.T, order=order, mode="constant", cval=0.0)
    return Volume(data=data.reshape(grid.shape).astype(np.float32),
                  affine=grid.affine, name=vol.name + "_resampled")
