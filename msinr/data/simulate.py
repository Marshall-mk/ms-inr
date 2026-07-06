"""Simulate multi-stack thick-slice acquisitions from an isotropic HR volume.

For each requested orientation (default: 3 orthogonal -- axial, coronal, sagittal)
we build a low-through-plane-resolution grid aligned to the world axes, optionally
apply a small rigid motion, then form each slice voxel as the anisotropic-Gaussian
PSF-weighted average of the HR volume sampled along the slice-select direction,
finally adding Rician noise. Output stacks carry correct affines (so NeSVoR can
ingest them) plus a JSON sidecar recording the exact simulation parameters and the
applied motion (ground truth for a later registration experiment).

Run standalone:
    python -m msinr.data.simulate --input HR.nii.gz --out data/sub01 \
        --config configs/simulate/default.yaml [--seed 0]
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import yaml
from scipy.ndimage import map_coordinates

from ..common.contracts import Volume, Stack
from ..common import io as mio
from ..common.geometry import apply_affine, rigid_matrix, grid_corners_world
from ..forward.multistack import gaussian_psf_local, stack_world_basis, psf_world_offsets

# voxel-axis -> world-axis permutation for each orientation; slice axis is voxel k.
ORIENTATIONS = {
    "axial":    (0, 1, 2),   # i->x, j->y, k->z (through = z)
    "coronal":  (0, 2, 1),   # i->x, j->z, k->y (through = y)
    "sagittal": (1, 2, 0),   # i->y, j->z, k->x (through = x)
}


def sample_volume_world(vol: Volume, world_pts: np.ndarray, order: int = 1) -> np.ndarray:
    """Trilinear-sample ``vol`` at world coordinates (P,3) -> (P,)."""
    inv = np.linalg.inv(vol.affine)
    vox = apply_affine(inv, world_pts)
    return map_coordinates(vol.data, vox.T, order=order, mode="constant", cval=0.0)


def _base_affine(orientation: str, in_plane: float, thickness: float,
                 world_min: np.ndarray) -> np.ndarray:
    """World-axis-aligned voxel->world affine for a stack of given orientation."""
    perm = ORIENTATIONS[orientation]
    spac = [in_plane, in_plane, thickness]           # along voxel i, j, k
    A = np.eye(4)
    A[:3, :3] = 0.0
    for vox_axis, w_axis in enumerate(perm):
        A[w_axis, vox_axis] = spac[vox_axis]
    A[:3, 3] = world_min
    return A


def simulate_stack(hr: Volume, orientation: str, in_plane: float, thickness: float,
                   psf_cfg: dict, motion_cfg: dict, snr: float,
                   rng: np.random.Generator) -> Stack:
    perm = ORIENTATIONS[orientation]
    corners = grid_corners_world(hr.shape, hr.affine)
    world_min, world_max = corners.min(0), corners.max(0)
    world_size = world_max - world_min
    fov_center = 0.5 * (world_min + world_max)

    A_base = _base_affine(orientation, in_plane, thickness, world_min)
    spac = np.array([in_plane, in_plane, thickness])
    n_vox = [int(np.ceil(world_size[perm[a]] / spac[a])) for a in range(3)]

    # optional rigid motion, rotating about the FOV centre
    M = np.eye(4)
    if motion_cfg.get("enabled", False):
        rot = rng.uniform(-1, 1, 3) * motion_cfg.get("max_rot_deg", 0.0)
        trans = rng.uniform(-1, 1, 3) * motion_cfg.get("max_trans_mm", 0.0)
        M = rigid_matrix(rot, trans, center=fov_center)
    A = M @ A_base

    # LR voxel world coordinates
    ii, jj, kk = np.meshgrid(*[np.arange(n) for n in n_vox], indexing="ij")
    vox = np.stack([ii.ravel(), jj.ravel(), kk.ravel()], -1).astype(np.float64)
    world = apply_affine(A, vox)                                    # (P,3)

    # PSF offsets in world frame
    local_off, weights = gaussian_psf_local(
        thickness, in_plane=(in_plane, in_plane),
        n_through=psf_cfg.get("n_through", 7), n_in=psf_cfg.get("n_in", 1),
        extent_sigma=psf_cfg.get("extent_sigma", 1.5),
        mode=psf_cfg.get("mode", "gaussian"))
    world_off = psf_world_offsets(local_off, stack_world_basis(A))  # (M,3)

    # accumulate PSF-weighted samples
    acc = np.zeros(world.shape[0], np.float64)
    for o, w in zip(world_off, weights):
        acc += w * sample_volume_world(hr, world + o)
    data = acc.reshape(n_vox)

    data = _add_rician_noise(data, snr, rng)

    meta = {
        "orientation": orientation, "in_plane_mm": in_plane, "thickness_mm": thickness,
        "psf": psf_cfg, "snr": snr,
        "motion": {"matrix": M.tolist(),
                   "enabled": bool(motion_cfg.get("enabled", False))},
        "hr_shape": list(hr.shape), "hr_affine": hr.affine.tolist(),
    }
    return Stack(data=data.astype(np.float32), affine=A, name=orientation,
                 slice_axis=2, meta=meta)


def _add_rician_noise(data: np.ndarray, snr: float, rng: np.random.Generator):
    if not snr or snr <= 0:
        return data
    ref = float(data[data > 0].mean()) if np.any(data > 0) else float(data.mean())
    sigma = ref / snr
    n1 = rng.normal(0, sigma, data.shape)
    n2 = rng.normal(0, sigma, data.shape)
    return np.sqrt((data + n1) ** 2 + n2 ** 2)


def simulate(hr: Volume, cfg: dict, seed: int = 0) -> list[Stack]:
    rng = np.random.default_rng(seed)
    stacks = []
    for spec in cfg["stacks"]:
        stacks.append(simulate_stack(
            hr, spec["orientation"],
            in_plane=spec.get("in_plane_mm", cfg.get("in_plane_mm", 1.0)),
            thickness=spec.get("thickness_mm", cfg.get("thickness_mm", 2.0)),
            psf_cfg=cfg.get("psf", {}), motion_cfg=cfg.get("motion", {}),
            snr=cfg.get("snr", 0.0), rng=rng))
    return stacks


def main():
    ap = argparse.ArgumentParser(description="Simulate multi-stack MRI from an isotropic HR volume.")
    ap.add_argument("--input", required=True, help="Isotropic HR NIfTI.")
    ap.add_argument("--out", required=True, help="Output directory for stacks + GT copy.")
    ap.add_argument("--config", required=True, help="Simulation YAML config.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    hr = mio.load_volume(args.input, name="gt")
    os.makedirs(args.out, exist_ok=True)
    mio.save_volume(hr, os.path.join(args.out, "gt.nii.gz"))
    stacks = simulate(hr, cfg, seed=args.seed)
    for i, st in enumerate(stacks):
        mio.save_stack(st, os.path.join(args.out, f"stack_{i:02d}_{st.name}.nii.gz"))
    print(f"Wrote GT + {len(stacks)} stacks to {args.out}")
    for st in stacks:
        print(f"  {st.name:9s} shape={st.shape} spacing={np.round(st.spacing,2)} "
              f"thickness={st.thickness:.1f}mm")


if __name__ == "__main__":
    main()
