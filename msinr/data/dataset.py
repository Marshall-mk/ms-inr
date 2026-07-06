"""Turn observed stacks into training tensors for INR reconstruction.

Each stack becomes a ``StackTensors`` holding: world coordinates of every
(optionally foreground) slice voxel, the observed intensities, and the stack's
PSF world offsets + weights. ``MultiStackSampler`` draws a balanced minibatch
across stacks each step.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..common.contracts import Stack, Volume, GridSpec
from ..common.geometry import apply_affine, grid_corners_world
from ..forward.multistack import gaussian_psf_local, stack_world_basis, psf_world_offsets


@dataclass
class StackTensors:
    coords_world: torch.Tensor   # (N,3)
    values: torch.Tensor         # (N,)
    offsets_world: torch.Tensor  # (M,3)
    weights: torch.Tensor        # (M,)
    name: str


def _psf_from_meta(stack: Stack, psf_override: dict | None):
    """Reconstruct the PSF used for this stack (from sidecar meta or override)."""
    meta = stack.meta or {}
    cfg = dict(meta.get("psf", {}))
    if psf_override:
        cfg.update(psf_override)
    ip = stack.spacing.copy()
    ip_vals = [float(ip[a]) for a in range(3) if a != stack.slice_axis]
    local, w = gaussian_psf_local(
        stack.thickness, in_plane=tuple(ip_vals),
        n_through=cfg.get("n_through", 7), n_in=cfg.get("n_in", 1),
        extent_sigma=cfg.get("extent_sigma", 1.5), mode=cfg.get("mode", "gaussian"))
    world_off = psf_world_offsets(local, stack_world_basis(stack.affine, stack.slice_axis))
    return world_off, w


def prepare_stack_tensors(stacks: list[Stack], device="cuda", dtype=torch.float32,
                          foreground_only: bool = True, psf_override: dict | None = None):
    out = []
    for st in stacks:
        shape = st.shape
        ii, jj, kk = np.meshgrid(*[np.arange(n) for n in shape], indexing="ij")
        vox = np.stack([ii.ravel(), jj.ravel(), kk.ravel()], -1).astype(np.float64)
        vals = st.data.ravel().astype(np.float32)
        if foreground_only:
            keep = vals > 0
            vox, vals = vox[keep], vals[keep]
        world = apply_affine(st.affine, vox)
        off, w = _psf_from_meta(st, psf_override)
        out.append(StackTensors(
            coords_world=torch.as_tensor(world, dtype=dtype, device=device),
            values=torch.as_tensor(vals, dtype=dtype, device=device),
            offsets_world=torch.as_tensor(off, dtype=dtype, device=device),
            weights=torch.as_tensor(w, dtype=dtype, device=device),
            name=st.name))
    return out


def _quantile(t: torch.Tensor, q: float) -> float:
    """Robust quantile that avoids torch.quantile's element-count limit."""
    n = t.numel()
    if n > 4_000_000:
        idx = torch.randint(0, n, (4_000_000,), device=t.device)
        t = t[idx]
    v = torch.quantile(t.float(), q).item()
    return v if v > 1e-8 else max(float(t.max()), 1e-8)


def normalize_stack_tensors(stack_tensors, mode: str = "global", q: float = 0.99) -> float:
    """Scale observed intensities to ~[0,1] in-place; return the factor to multiply
    the reconstruction by to return to input units.

    - 'global':    divide all stacks by a shared percentile (simulated data from one
                   source -> stacks share an intensity scale). Output rescaled back.
    - 'per_stack': divide each stack by its own percentile (real multi-scanner data
                   with differing scales). Output stays in normalized units (1.0).
    - 'none':      no change.
    """
    if mode == "none":
        return 1.0
    if mode == "per_stack":
        for s in stack_tensors:
            s.values = s.values / _quantile(s.values, q)
        return 1.0
    # global
    allv = torch.cat([s.values for s in stack_tensors])
    scale = _quantile(allv, q)
    for s in stack_tensors:
        s.values = s.values / scale
    return scale


class MultiStackSampler:
    """Draws a balanced random minibatch across all stacks each ``step``."""

    def __init__(self, stack_tensors: list[StackTensors], batch_per_stack: int = 4096,
                 generator: torch.Generator | None = None):
        self.stacks = stack_tensors
        self.bps = batch_per_stack
        self.g = generator

    def sample(self):
        """Return a list of (coords, offsets, weights, targets) per stack."""
        out = []
        for s in self.stacks:
            n = s.values.shape[0]
            idx = torch.randint(0, n, (min(self.bps, n),), device=s.values.device,
                                generator=self.g)
            out.append((s.coords_world[idx], s.offsets_world, s.weights, s.values[idx]))
        return out


# ---------------------------------------------------------------------------
# reconstruction grid + normalizer
# ---------------------------------------------------------------------------
def recon_grid_from_stacks(stacks: list[Stack], iso_mm: float = 1.0) -> GridSpec:
    """World-axis-aligned isotropic grid covering the union of all stack FOVs."""
    allc = np.concatenate([grid_corners_world(s.shape, s.affine) for s in stacks], 0)
    lo, hi = allc.min(0), allc.max(0)
    shape = tuple(int(np.ceil((hi[a] - lo[a]) / iso_mm)) + 1 for a in range(3))
    affine = np.eye(4)
    affine[0, 0] = affine[1, 1] = affine[2, 2] = iso_mm
    affine[:3, 3] = lo
    return GridSpec(shape=shape, affine=affine)
