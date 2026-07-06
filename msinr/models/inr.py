"""INR architectures for 3D coordinate -> intensity, adapted from the reference
implementation of McGinnis et al. 2025 (arXiv:2512.14366,
github.com/jqmcginnis/stable_rank_inrs, models.py).

Trimmed to the variants relevant for MRI SRR (ReLU-MLP, ReLU Fourier-features,
ReLU positional-encoding, SIREN) and defaulted to input_dim=3, output_dim=1.
The rank-tracking machinery (``matrix_info`` -> stable rank, effective rank,
spectral norm, condition number) is preserved verbatim so we can reproduce the
paper's diagnostics in the SRR setting.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn


# ---------------------------------------------------------------------------
# rank / spectral diagnostics (verbatim from the reference)
# ---------------------------------------------------------------------------
def matrix_info(weight, eps: float = 1e-12) -> dict:
    with torch.no_grad():
        w = weight.detach()
        sv = torch.linalg.svdvals(w) + eps
        spectral = torch.max(sv)
        fro_sq = torch.sum(sv ** 2)
        stable_rank = fro_sq / (spectral ** 2)
        p = (sv ** 2) / fro_sq
        eff_rank = torch.exp(-torch.sum(p * torch.log(p)))
        cond = spectral / torch.min(sv)
    return {
        "spectral_norm": spectral.item(),
        "frobenius_norm": torch.sqrt(fro_sq).item(),
        "stable_rank": stable_rank.item(),
        "effective_rank": eff_rank.item(),
        "condition_number": cond.item(),
    }


class SinusoidalActivation(nn.Module):
    def __init__(self, omega: float = 30.0):
        super().__init__()
        self.register_buffer("omega", torch.tensor(float(omega)))

    def forward(self, x):
        return torch.sin(self.omega * x)


class ActivatedLinear(nn.Module):
    """Linear layer (+ optional activation) exposing per-layer rank diagnostics."""

    def __init__(self, in_features, out_features, activation=None,
                 layer_type="hidden", siren_omega=None):
        super().__init__()
        self.activation = activation
        self.in_features = in_features
        self.layer_type = layer_type
        self.linear = nn.Linear(in_features, out_features)
        if siren_omega is not None:
            self._siren_init(layer_type, siren_omega)

    def _siren_init(self, layer_type, omega):
        with torch.no_grad():
            if layer_type == "first":
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            else:
                b = np.sqrt(6 / self.in_features) / omega
                self.linear.weight.uniform_(-b, b)

    def forward(self, x):
        x = self.linear(x)
        return x if self.activation is None else self.activation(x)

    def get_info(self):
        return matrix_info(self.linear.weight)


class _RankTrackedMLP(nn.Module):
    """Base class providing the rank-diagnostics API over ``self.mlp``."""

    def get_layer_infos(self):
        return [m.get_info() for m in self.mlp if isinstance(m, ActivatedLinear)]

    def get_detailed_matrix_info(self):
        return {"layer_infos": self.get_layer_infos()}


# ---------------------------------------------------------------------------
# architectures
# ---------------------------------------------------------------------------
class ReluMLP(_RankTrackedMLP):
    def __init__(self, input_dim=3, hidden_dim=256, output_dim=1, num_layers=4):
        super().__init__()
        layers = [ActivatedLinear(input_dim, hidden_dim, nn.ReLU())]
        for _ in range(num_layers - 2):
            layers.append(ActivatedLinear(hidden_dim, hidden_dim, nn.ReLU()))
        layers.append(ActivatedLinear(hidden_dim, output_dim, activation=None))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class ReluFFN(_RankTrackedMLP):
    """ReLU MLP on random Fourier features (fixed Gaussian B)."""

    def __init__(self, input_dim=3, mapping_size=128, hidden_dim=256,
                 output_dim=1, num_layers=4, sigma=5.0):
        super().__init__()
        B = torch.randn(mapping_size, input_dim) * sigma
        self.register_buffer("B", B)
        layers = [ActivatedLinear(2 * mapping_size, hidden_dim, nn.ReLU())]
        for _ in range(num_layers - 2):
            layers.append(ActivatedLinear(hidden_dim, hidden_dim, nn.ReLU()))
        layers.append(ActivatedLinear(hidden_dim, output_dim, activation=None))
        self.mlp = nn.Sequential(*layers)

    def _encode(self, x):
        proj = 2 * math.pi * (x @ self.B.T)
        return torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)

    def forward(self, x):
        return self.mlp(self._encode(x))


class ReluPosEncoding(_RankTrackedMLP):
    """ReLU MLP on NeRF-style positional encoding (deterministic frequencies)."""

    def __init__(self, input_dim=3, num_freqs=10, hidden_dim=256,
                 output_dim=1, num_layers=4):
        super().__init__()
        self.num_freqs = num_freqs
        mlp_in = input_dim * 2 * num_freqs
        layers = [ActivatedLinear(mlp_in, hidden_dim, nn.ReLU())]
        for _ in range(num_layers - 2):
            layers.append(ActivatedLinear(hidden_dim, hidden_dim, nn.ReLU()))
        layers.append(ActivatedLinear(hidden_dim, output_dim, activation=None))
        self.mlp = nn.Sequential(*layers)

    def _encode(self, x):
        freqs = 2 ** torch.arange(self.num_freqs, device=x.device, dtype=x.dtype)
        xb = x.unsqueeze(-1) * freqs * math.pi          # (..., D, F)
        enc = torch.stack([torch.sin(xb), torch.cos(xb)], dim=-1)
        return enc.flatten(start_dim=-3)                # (..., D*F*2)

    def forward(self, x):
        return self.mlp(self._encode(x))


class SirenMLP(_RankTrackedMLP):
    def __init__(self, input_dim=3, hidden_dim=256, output_dim=1,
                 num_layers=4, omega=30.0):
        super().__init__()
        omegas = omega if isinstance(omega, (list, tuple)) else [omega] * (num_layers - 1)
        layers = [ActivatedLinear(input_dim, hidden_dim, SinusoidalActivation(omegas[0]),
                                  layer_type="first", siren_omega=omegas[0])]
        for i in range(num_layers - 2):
            layers.append(ActivatedLinear(hidden_dim, hidden_dim,
                                          SinusoidalActivation(omegas[i + 1]),
                                          layer_type="hidden", siren_omega=omegas[i + 1]))
        layers.append(ActivatedLinear(hidden_dim, output_dim, activation=None,
                                      layer_type="last", siren_omega=omegas[-1]))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


_REGISTRY = {
    "relu_mlp": ReluMLP,
    "relu_ffn": ReluFFN,
    "relu_pos_enc": ReluPosEncoding,
    "siren_mlp": SirenMLP,
}


def build_inr(name: str, **kwargs) -> nn.Module:
    """Factory: ``build_inr("relu_ffn", hidden_dim=256, num_layers=4, sigma=6.0)``."""
    if name not in _REGISTRY:
        raise ValueError(f"unknown INR '{name}', options: {list(_REGISTRY)}")
    cls = _REGISTRY[name]
    valid = cls.__init__.__code__.co_varnames
    kw = {k: v for k, v in kwargs.items() if k in valid}
    return cls(**kw)
