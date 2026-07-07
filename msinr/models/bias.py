"""Per-stack intensity model for real (low-field) data.

Different stacks disagree on intensity (coil sensitivity, B1 inhomogeneity, low-field
bias). Without modelling this, the coordinate field is forced to reconcile
inconsistent observations and ends up fitting them as high-frequency noise. We model
each stack's acquisition as a smooth multiplicative field:

    y_k(p) ~= exp(b_k(p)) * (PSF-convolved f(p))

where ``f`` is the clean latent volume (what we output) and ``b_k`` is a low-order
polynomial in normalized coordinates (degree-0 term = per-stack log-gain). Fitting
``b_k`` absorbs the inter-stack inconsistency so ``f`` stays clean. Standard in
NeSVoR/SVRTK.
"""
from __future__ import annotations

import torch
from torch import nn


def _poly_exponents(degree: int):
    """3D monomial exponents (a,b,c) with 1 <= a+b+c <= degree (constant excluded)."""
    terms = []
    for total in range(1, degree + 1):
        for a in range(total + 1):
            for b in range(total - a + 1):
                terms.append((a, b, total - a - b))
    return terms


class PerStackBias(nn.Module):
    """Multiplicative per-stack bias factor. ``mode`` in {'gain', 'poly'}."""

    def __init__(self, n_stacks: int, degree: int = 2, mode: str = "poly"):
        super().__init__()
        self.mode = mode
        self.log_gain = nn.Parameter(torch.zeros(n_stacks))     # degree-0 (per-stack gain)
        if mode == "poly":
            self.exps = _poly_exponents(degree)
            self.coef = nn.Parameter(torch.zeros(n_stacks, len(self.exps)))

    def factor(self, k: int, coords_norm: torch.Tensor) -> torch.Tensor:
        """Multiplicative factor for stack ``k`` at normalized coords (N,3) -> (N,)."""
        logb = self.log_gain[k]
        if self.mode == "poly":
            x, y, z = coords_norm[:, 0], coords_norm[:, 1], coords_norm[:, 2]
            basis = torch.stack([(x ** a) * (y ** b) * (z ** c)
                                 for (a, b, c) in self.exps], dim=-1)   # (N, T)
            logb = logb + (basis * self.coef[k]).sum(-1)
        return torch.exp(logb)
