"""Geometry helpers: rigid transforms, affines, world<->normalized coordinates.

The INR operates on **normalized coordinates in [-1, 1]^3** aligned to the target
reconstruction grid's world bounding box. ``CoordNormalizer`` is the single source
of truth for that mapping and MUST be shared between training (sampling observed
slices) and inference (sampling the HR grid), otherwise the field is misaligned.
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Rigid transforms
# ---------------------------------------------------------------------------
def euler_to_rotmat(rx: float, ry: float, rz: float, degrees: bool = True) -> np.ndarray:
    """Rotation matrix from intrinsic X-Y-Z Euler angles."""
    if degrees:
        rx, ry, rz = np.deg2rad([rx, ry, rz])
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def rigid_matrix(rotations_deg=(0, 0, 0), translations_mm=(0, 0, 0),
                 center=(0, 0, 0)) -> np.ndarray:
    """4x4 homogeneous rigid transform, rotating about ``center`` (world mm)."""
    R = euler_to_rotmat(*rotations_deg, degrees=True)
    t = np.asarray(translations_mm, dtype=np.float64)
    c = np.asarray(center, dtype=np.float64)
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = c - R @ c + t
    return M


def apply_affine(affine: np.ndarray, coords: np.ndarray) -> np.ndarray:
    """Apply a 4x4 affine to an (N, 3) array of coordinates -> (N, 3)."""
    coords = np.asarray(coords, dtype=np.float64)
    return coords @ affine[:3, :3].T + affine[:3, 3]


def voxel_grid_world(shape, affine: np.ndarray) -> np.ndarray:
    """World coordinates (X*Y*Z, 3) of all voxel centres of a grid, C-order."""
    xs, ys, zs = [np.arange(s) for s in shape]
    ii, jj, kk = np.meshgrid(xs, ys, zs, indexing="ij")
    vox = np.stack([ii.ravel(), jj.ravel(), kk.ravel()], axis=-1).astype(np.float64)
    return apply_affine(affine, vox)


def grid_corners_world(shape, affine: np.ndarray) -> np.ndarray:
    """World coordinates of the 8 corners of the voxel index box [-0.5, N-0.5]."""
    lo = -0.5
    corners = np.array([[lo if b == 0 else s - 0.5 for b, s in zip(bits, shape)]
                        for bits in np.ndindex(2, 2, 2)], dtype=np.float64)
    return apply_affine(affine, corners)


# ---------------------------------------------------------------------------
# World <-> normalized [-1, 1] coordinates
# ---------------------------------------------------------------------------
class CoordNormalizer:
    """Maps world mm coordinates to/from [-1, 1]^3 using a grid's world bbox.

    Built once from the reconstruction ``GridSpec`` (or GT volume) and reused
    everywhere so training samples and inference samples live in the same frame.
    """

    def __init__(self, center: np.ndarray, half_extent: np.ndarray):
        self.center = np.asarray(center, dtype=np.float64)
        # guard against a zero-thickness axis
        self.half_extent = np.maximum(np.asarray(half_extent, dtype=np.float64), 1e-6)

    @classmethod
    def from_grid(cls, shape, affine: np.ndarray, pad_frac: float = 0.0) -> "CoordNormalizer":
        corners = grid_corners_world(shape, affine)
        lo, hi = corners.min(0), corners.max(0)
        center = 0.5 * (lo + hi)
        half = 0.5 * (hi - lo)
        half = half * (1.0 + pad_frac)
        return cls(center, half)

    def to_norm(self, world: np.ndarray) -> np.ndarray:
        return (np.asarray(world, dtype=np.float64) - self.center) / self.half_extent

    def to_world(self, norm: np.ndarray) -> np.ndarray:
        return np.asarray(norm, dtype=np.float64) * self.half_extent + self.center

    def as_dict(self) -> dict:
        return {"center": self.center.tolist(), "half_extent": self.half_extent.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "CoordNormalizer":
        return cls(np.array(d["center"]), np.array(d["half_extent"]))
