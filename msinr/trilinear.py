"""Trilinear-averaging baseline: resample every stack onto the reconstruction grid
by trilinear interpolation and average where stacks overlap. A registration-free,
learning-free lower bound (what you get without any real super-resolution)."""
from __future__ import annotations

import numpy as np

from .common.contracts import Volume, GridSpec, ReconResult
from .common.profiling import Profiler
from .common.resample import resample_to_grid
from .data.dataset import recon_grid_from_stacks


def reconstruct_trilinear(stacks, gt: Volume | None, cfg: dict) -> ReconResult:
    grid = GridSpec.from_volume(gt) if gt is not None \
        else recon_grid_from_stacks(stacks, iso_mm=cfg.get("iso_mm", 1.0))
    from .common.roi import maybe_crop_to_roi
    grid, roi = maybe_crop_to_roi(grid, cfg)   # restrict to brain ROI if given
    mode = cfg.get("normalize_stacks", "global")

    prof = Profiler("cpu")
    acc = np.zeros(grid.shape, np.float64)
    cnt = np.zeros(grid.shape, np.float64)
    with prof.section("reconstruct"):
        for st in stacks:
            rs = resample_to_grid(Volume(st.data, st.affine, st.name), grid).data
            if mode == "per_stack":
                p = np.percentile(st.data[st.data > 0], 99) if np.any(st.data > 0) else 1.0
                rs = rs / (p + 1e-8)
            m = rs > 0
            acc[m] += rs[m]
            cnt[m] += 1.0
    recon = np.where(cnt > 0, acc / np.maximum(cnt, 1.0), 0.0).astype(np.float32)
    if roi is not None:                        # output only the brain-masked region
        from .common.roi import mask_on_grid
        recon = recon * mask_on_grid(grid, roi)

    prof.sections["inference"] = {"seconds": 0.0}
    prof.add("num_parameters", 0)
    vol = Volume(data=recon, affine=grid.affine, name="recon_trilinear")
    return ReconResult(volume=vol, method="trilinear", config=dict(cfg),
                       profile=prof.summary())
