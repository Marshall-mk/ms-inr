"""Classical least-squares super-resolution reconstruction (LS-SRR).

Solves, on the discrete HR grid ``x``,

    min_x  sum_k || A_k x - y_k ||^2  +  lambda ||x||^2   (Tikhonov)

where ``A_k`` is the SAME anisotropic-Gaussian PSF + slice-sampling operator used
by the INR forward model, here materialized as a sparse (samples x voxels)
trilinear-interpolation matrix. The normal equations are solved with conjugate
gradient. This is a strong, learning-free baseline (IREM/NiftyMIC family) and a
clean point of comparison for the INR methods. Runs on CPU via scipy (safe,
deterministic); device is recorded in the profile.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, cg

from .common.contracts import Volume, GridSpec, ReconResult
from .common.geometry import apply_affine
from .common.profiling import Profiler
from .data.dataset import recon_grid_from_stacks
from .forward.multistack import gaussian_psf_local, stack_world_basis, psf_world_offsets


def _trilinear_rows(frac_vox: np.ndarray, shape, sample_ids, base_weight):
    """COO (row, col, val) contributions for trilinear sampling at fractional
    voxel coords ``frac_vox`` (P,3) on a grid of ``shape``. ``base_weight`` (P,)
    scales each sample (the PSF weight)."""
    base = np.floor(frac_vox).astype(np.int64)
    df = frac_vox - base
    rows, cols, vals = [], [], []
    X, Y, Z = shape
    for cx in (0, 1):
        wx = df[:, 0] if cx else 1 - df[:, 0]
        ix = base[:, 0] + cx
        for cy in (0, 1):
            wy = df[:, 1] if cy else 1 - df[:, 1]
            iy = base[:, 1] + cy
            for cz in (0, 1):
                wz = df[:, 2] if cz else 1 - df[:, 2]
                iz = base[:, 2] + cz
                w = wx * wy * wz * base_weight
                valid = (ix >= 0) & (ix < X) & (iy >= 0) & (iy < Y) & (iz >= 0) & (iz < Z)
                lin = (ix * Y + iy) * Z + iz
                rows.append(sample_ids[valid])
                cols.append(lin[valid])
                vals.append(w[valid])
    return np.concatenate(rows), np.concatenate(cols), np.concatenate(vals)


def build_operator(stacks, grid: GridSpec, foreground_only=True, psf_override=None):
    """Return (A, y): sparse forward matrix (S x V) and stacked observations (S,)."""
    inv = np.linalg.inv(grid.affine)
    V = int(np.prod(grid.shape))
    rows, cols, vals, ys = [], [], [], []
    s0 = 0
    for st in stacks:
        ii, jj, kk = np.meshgrid(*[np.arange(n) for n in st.shape], indexing="ij")
        vox = np.stack([ii.ravel(), jj.ravel(), kk.ravel()], -1).astype(np.float64)
        obs = st.data.ravel().astype(np.float64)
        if foreground_only:
            keep = obs > 0
            vox, obs = vox[keep], obs[keep]
        world = apply_affine(st.affine, vox)                       # (P,3)
        cfg = dict(st.meta.get("psf", {})); cfg.update(psf_override or {})
        ip = [float(st.spacing[a]) for a in range(3) if a != st.slice_axis]
        local, w = gaussian_psf_local(st.thickness, in_plane=tuple(ip),
                                      n_through=cfg.get("n_through", 7),
                                      n_in=cfg.get("n_in", 1),
                                      extent_sigma=cfg.get("extent_sigma", 1.5),
                                      mode=cfg.get("mode", "gaussian"))
        world_off = psf_world_offsets(local, stack_world_basis(st.affine, st.slice_axis))
        sample_ids = np.arange(world.shape[0]) + s0
        for o, wo in zip(world_off, w):
            frac = apply_affine(inv, world + o)
            r, c, v = _trilinear_rows(frac, grid.shape, sample_ids, np.full(world.shape[0], wo))
            rows.append(r); cols.append(c); vals.append(v)
        ys.append(obs)
        s0 += world.shape[0]
    S = s0
    # int32 indices + float32 values roughly halve peak memory (matters for the
    # large 512^2 real stacks, which otherwise OOM). V and nnz both fit in int32.
    r = np.concatenate(rows).astype(np.int32); rows.clear()
    c = np.concatenate(cols).astype(np.int32); cols.clear()
    v = np.concatenate(vals).astype(np.float32); vals.clear()
    A = sp.coo_matrix((v, (r, c)), shape=(S, V)).tocsr()
    del r, c, v
    return A, np.concatenate(ys)


def reconstruct_classical(stacks, gt: Volume | None, cfg: dict) -> ReconResult:
    grid = GridSpec.from_volume(gt) if gt is not None \
        else recon_grid_from_stacks(stacks, iso_mm=cfg.get("iso_mm", 1.0))
    lam = float(cfg.get("reg_lambda", 1e-1))
    maxiter = int(cfg.get("cg_maxiter", 200))
    tol = float(cfg.get("cg_tol", 1e-5))

    prof = Profiler("cpu")
    A, y = build_operator(stacks, grid,
                          foreground_only=cfg.get("foreground_only", True),
                          psf_override=cfg.get("psf"))
    V = A.shape[1]
    # normalize observations to ~[0,1] so reg_lambda is scale-consistent; rescale back
    scale = float(np.percentile(y, 99)) if cfg.get("normalize_stacks", "global") != "none" else 1.0
    scale = scale if scale > 1e-8 else 1.0
    y = y / scale
    At = A.T                       # CSC view sharing A's data (no extra copy)
    rhs = At @ y
    H = LinearOperator((V, V), matvec=lambda x: At @ (A @ x) + lam * x, dtype=np.float64)

    iters = {"n": 0}
    def _cb(_): iters["n"] += 1
    with prof.section("reconstruct"):
        x, info = cg(H, rhs, rtol=tol, maxiter=maxiter, callback=_cb)
    recon = (np.clip(x, 0, None) * scale).reshape(grid.shape).astype(np.float32)

    prof.add("num_parameters", V)
    prof.add("cg_iterations", iters["n"])
    prof.add("cg_converged", int(info == 0))
    prof.add("reg_lambda", lam)
    # for the classical solver "inference" is free (recon is already the grid)
    prof.sections["inference"] = {"seconds": 0.0}

    vol = Volume(data=recon, affine=grid.affine, name="recon_classical")
    return ReconResult(volume=vol, method="classical_srr", config=dict(cfg),
                       profile=prof.summary())
