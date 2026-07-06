"""Differentiable multi-stack acquisition forward model.

The latent isotropic volume is a continuous field ``f`` (an INR, or a sampled
grid). An observed slice value at world point ``p`` on stack ``k`` is the
PSF-weighted integral of ``f`` along the slice-select direction (and, optionally,
in-plane):

    y_hat_k(p) = sum_m  w_m * f( p + o_m )

where ``{o_m}`` are world-space offsets drawn from an anisotropic Gaussian PSF
(NeSVoR convention: through-plane FWHM = slice thickness, i.e. sigma =
thickness / 2.3552; in-plane FWHM ~= 1.2 x in-plane spacing) and ``{w_m}`` the
corresponding normalized weights. A "sampling-only" mode (``mode='delta'``,
IREM-style) uses a single zero offset.

The same operator is used three ways:
  * INR training  -> ``f`` evaluates the network at PSF-expanded coordinates;
  * INR inference -> sample ``f`` on the HR grid (no PSF);
  * classical SRR -> ``A x`` applies the PSF+sampling to a discrete HR grid.
"""
from __future__ import annotations

import numpy as np
import torch

FWHM_TO_SIGMA = 1.0 / 2.3548200450309493   # 1 / (2*sqrt(2*ln2))


# ---------------------------------------------------------------------------
# PSF offset construction (world-agnostic local basis: [in0, in1, through])
# ---------------------------------------------------------------------------
def gaussian_psf_local(thickness: float, in_plane=(1.0, 1.0), *,
                       n_through: int = 7, n_in: int = 1,
                       extent_sigma: float = 1.5, mode: str = "gaussian"):
    """Local-frame PSF offsets (M,3) and weights (M,).

    Columns are ordered (in-plane-0, in-plane-1, through-plane). ``mode='delta'``
    returns a single zero-offset sample (IREM sampling-only model).
    """
    if mode == "delta":
        return np.zeros((1, 3), np.float64), np.ones(1, np.float64)

    def axis(sigma, n):
        if n <= 1 or sigma <= 0:
            return np.zeros(1), np.ones(1)
        s = np.linspace(-extent_sigma * sigma, extent_sigma * sigma, n)
        w = np.exp(-0.5 * (s / sigma) ** 2)
        return s, w

    st = thickness * FWHM_TO_SIGMA
    s_thr, w_thr = axis(st, n_through)
    s_i0, w_i0 = axis(in_plane[0] * 1.2 * FWHM_TO_SIGMA, n_in)
    s_i1, w_i1 = axis(in_plane[1] * 1.2 * FWHM_TO_SIGMA, n_in)

    offs, ws = [], []
    for a, wa in zip(s_i0, w_i0):
        for b, wb in zip(s_i1, w_i1):
            for c, wc in zip(s_thr, w_thr):
                offs.append((a, b, c))
                ws.append(wa * wb * wc)
    offs = np.asarray(offs, np.float64)
    ws = np.asarray(ws, np.float64)
    ws /= ws.sum()
    return offs, ws


def stack_world_basis(affine: np.ndarray, slice_axis: int = 2) -> np.ndarray:
    """Orthonormal-ish world directions for a stack's (in0, in1, through) axes.

    Columns are the normalized world directions of the two in-plane voxel axes
    and the slice-select axis, taken from the stack affine.
    """
    in_axes = [a for a in range(3) if a != slice_axis]
    cols = []
    for a in in_axes + [slice_axis]:
        v = affine[:3, a].astype(np.float64)
        cols.append(v / (np.linalg.norm(v) + 1e-12))
    return np.stack(cols, axis=1)   # (3,3): [in0, in1, through]


def psf_world_offsets(local_offsets: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Rotate local PSF offsets (M,3) into world offsets (M,3) via a (3,3) basis."""
    return local_offsets @ basis.T


# ---------------------------------------------------------------------------
# Forward operator over normalized coordinates
# ---------------------------------------------------------------------------
class MultiStackForward:
    """Evaluates a field at PSF-expanded, normalized coordinates.

    ``center`` / ``half_extent`` come from a ``CoordNormalizer`` (shared frame).
    """

    def __init__(self, center, half_extent, device="cuda", dtype=torch.float32):
        self.device = device
        self.dtype = dtype
        self.center = torch.as_tensor(center, dtype=dtype, device=device)
        self.half_extent = torch.as_tensor(half_extent, dtype=dtype, device=device)

    def to_norm(self, world: torch.Tensor) -> torch.Tensor:
        return (world - self.center) / self.half_extent

    def predict_batch(self, field, batches):
        """PSF-weighted prediction for several stacks with a SINGLE field evaluation.

        ``batches`` is a list of (coords_world (N,3), offsets_world (M,3),
        weights (M,)). For each observed sample the field is read at its PSF
        offsets and combined by ``weights``. All PSF-expanded, normalized
        coordinates across stacks are concatenated into one tensor so the
        (expensive) network forward runs once per step instead of once per stack.
        Returns a list of (N,) predictions, one per input stack.
        """
        all_pts, meta = [], []
        for coords_world, offsets_world, weights in batches:
            N, M = coords_world.shape[0], offsets_world.shape[0]
            pts = coords_world[:, None, :] + offsets_world[None, :, :]
            all_pts.append(self.to_norm(pts).reshape(N * M, 3))
            meta.append((N, M, weights))
        vals = field(torch.cat(all_pts, dim=0))                     # ONE forward
        out, idx = [], 0
        for N, M, weights in meta:
            v = vals[idx:idx + N * M].reshape(N, M)
            idx += N * M
            out.append((v * weights[None, :]).sum(dim=1))
        return out
