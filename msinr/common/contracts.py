"""Core data structures shared across the whole pipeline.

Everything is expressed in **world millimetre coordinates** via NIfTI-style 4x4
affines (voxel index -> world). Intensities are plain numpy arrays; torch tensors
live only inside the forward model and the INR/optimizers.

The *input contract* of every reconstruction method is a list of ``Stack`` (the
observed thick-slice acquisitions) plus an optional ground-truth ``Volume``. The
*output contract* is a ``ReconResult`` (a reconstructed ``Volume`` + metrics dict
+ profile dict), which each ``methods/<name>/run.py`` serialises to disk.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional
import json

import numpy as np


@dataclass
class Volume:
    """An isotropic (or arbitrary) 3D volume with a voxel->world affine."""

    data: np.ndarray            # (X, Y, Z) float32
    affine: np.ndarray          # (4, 4) voxel index -> world (mm)
    name: str = "volume"

    def __post_init__(self):
        self.data = np.asarray(self.data, dtype=np.float32)
        self.affine = np.asarray(self.affine, dtype=np.float64)
        assert self.data.ndim == 3, f"expected 3D volume, got {self.data.shape}"
        assert self.affine.shape == (4, 4)

    @property
    def shape(self):
        return self.data.shape

    @property
    def spacing(self) -> np.ndarray:
        """Voxel spacing (mm) along each axis, from affine column norms."""
        return np.linalg.norm(self.affine[:3, :3], axis=0)


@dataclass
class Stack:
    """A single thick-slice acquisition.

    Convention: axis ``slice_axis`` (default 2, the last) is the through-plane
    / slice-select direction with thick spacing; the other two axes are the
    high-resolution in-plane axes. The affine already encodes spacing, orientation
    and any applied rigid motion, so world placement is fully determined by it.
    """

    data: np.ndarray            # (Ni, Nj, Nslices) float32
    affine: np.ndarray          # (4, 4) voxel -> world (mm)
    name: str = "stack"
    slice_axis: int = 2
    # provenance / simulation metadata (optional; helps reproducibility & debugging)
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.data = np.asarray(self.data, dtype=np.float32)
        self.affine = np.asarray(self.affine, dtype=np.float64)
        assert self.data.ndim == 3
        assert self.affine.shape == (4, 4)

    @property
    def shape(self):
        return self.data.shape

    @property
    def spacing(self) -> np.ndarray:
        return np.linalg.norm(self.affine[:3, :3], axis=0)

    @property
    def slice_normal(self) -> np.ndarray:
        """Unit world-space direction of the slice-select axis."""
        n = self.affine[:3, self.slice_axis].astype(np.float64)
        return n / (np.linalg.norm(n) + 1e-12)

    @property
    def thickness(self) -> float:
        """Slice thickness (mm) = spacing along the slice axis."""
        return float(self.spacing[self.slice_axis])


@dataclass
class GridSpec:
    """Describes a target reconstruction grid: shape + voxel->world affine."""

    shape: tuple            # (X, Y, Z)
    affine: np.ndarray      # (4, 4)

    def __post_init__(self):
        self.shape = tuple(int(s) for s in self.shape)
        self.affine = np.asarray(self.affine, dtype=np.float64)

    @classmethod
    def from_volume(cls, vol: Volume) -> "GridSpec":
        return cls(shape=vol.shape, affine=vol.affine.copy())


@dataclass
class ReconResult:
    """Output contract of a reconstruction method."""

    volume: Volume
    metrics: dict = field(default_factory=dict)   # quality metrics vs GT (if any)
    profile: dict = field(default_factory=dict)   # compute/timing metrics
    method: str = "unknown"
    config: dict = field(default_factory=dict)

    def save_sidecars(self, metrics_path: str, profile_path: str):
        """Write metrics.json and profile.json (numpy-safe)."""
        with open(metrics_path, "w") as f:
            json.dump(_jsonable({"method": self.method, **self.metrics}), f, indent=2)
        with open(profile_path, "w") as f:
            json.dump(_jsonable({"method": self.method, **self.profile}), f, indent=2)


def _jsonable(obj):
    """Recursively convert numpy scalars/arrays to plain python for json.dump."""
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj
