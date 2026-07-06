"""Optimizer construction for INRs, including the Muon parameter split.

Muon (Jordan et al.) is applied to the **hidden 2D weight matrices only**; the
first/last linear layers and all biases/vectors go to an auxiliary Adam, via
``SingleDeviceMuonWithAuxAdam``. This mirrors the recipe in
the reference fit_sisr.py and README (github.com/jqmcginnis/stable_rank_inrs).

``muon`` is imported lazily so the rest of the pipeline works without it
installed (only the proposed method needs it):
    pip install "git+https://github.com/KellerJordan/Muon"
"""
from __future__ import annotations

import torch


def split_muon_params(model):
    """Return (muon_params, adam_params).

    Collects 2D weight matrices in registration order; first & last -> Adam,
    hidden -> Muon; everything else (biases, buffers-as-params, 1D) -> Adam.
    """
    matrices, adam_params = [], []
    for _, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 2 and p.size(0) > 1 and p.size(1) > 1:
            matrices.append(p)
        else:
            adam_params.append(p)
    if len(matrices) < 3:
        # too shallow to have "hidden" matrices -> everything to Adam
        return [], adam_params + matrices
    adam_params.append(matrices[0])     # first layer
    adam_params.append(matrices[-1])    # last layer
    muon_params = matrices[1:-1]        # hidden layers
    return muon_params, adam_params


def build_optimizer(model, optimizer: str = "muon", *, lr: float = 1e-2,
                    muon_lr: float = 1e-1, weight_decay: float = 0.0,
                    muon_weight_decay: float = 0.0):
    """Build the optimizer. ``optimizer`` in {"muon", "adam"}."""
    optimizer = optimizer.lower()
    if optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    if optimizer == "muon":
        try:
            from muon import SingleDeviceMuonWithAuxAdam
        except ImportError as e:
            raise ImportError(
                "Muon not installed. Run: "
                'pip install "git+https://github.com/KellerJordan/Muon"'
            ) from e
        muon_params, adam_params = split_muon_params(model)
        if not muon_params:
            raise ValueError("No hidden weight matrices found for Muon; "
                             "use a deeper network or optimizer='adam'.")
        groups = [
            dict(params=muon_params, use_muon=True, lr=muon_lr,
                 weight_decay=muon_weight_decay),
            dict(params=adam_params, use_muon=False, lr=lr, betas=(0.9, 0.999),
                 weight_decay=weight_decay),
        ]
        return SingleDeviceMuonWithAuxAdam(groups)

    raise ValueError(f"unknown optimizer '{optimizer}'")
