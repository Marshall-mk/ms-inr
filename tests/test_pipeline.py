"""Fast CPU sanity tests for geometry, PSF, forward operator, and simulation.

    conda run -n dev python -m pytest tests/ -q
Kept lightweight (no GPU, tiny volumes) so they run in seconds.
"""
import numpy as np

from msinr.common.contracts import Volume, GridSpec
from msinr.common.geometry import (rigid_matrix, CoordNormalizer, apply_affine)
from msinr.forward.multistack import gaussian_psf_local, FWHM_TO_SIGMA
from msinr.data.simulate import simulate
from msinr.classical import build_operator


def _phantom(n=24):
    lin = np.linspace(-1, 1, n)
    x, y, z = np.meshgrid(lin, lin, lin, indexing="ij")
    data = np.zeros((n, n, n), np.float32)
    data[(x**2 + y**2 + z**2) < 0.7**2] = 1.0
    affine = np.eye(4)
    affine[:3, 3] = -0.5 * (n - 1)
    return Volume(data, affine, "gt")


def test_rigid_matrix_is_rigid():
    M = rigid_matrix((5, -3, 7), (1, 2, -1), center=(3, 3, 3))
    R = M[:3, :3]
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-10)
    assert np.isclose(np.linalg.det(R), 1.0, atol=1e-10)


def test_normalizer_roundtrip():
    vol = _phantom()
    nrm = CoordNormalizer.from_grid(vol.shape, vol.affine)
    w = np.random.default_rng(0).uniform(-10, 10, (100, 3))
    assert np.allclose(nrm.to_world(nrm.to_norm(w)), w, atol=1e-8)


def test_psf_weights_sum_to_one():
    off, w = gaussian_psf_local(2.0, (1.0, 1.0), n_through=7, n_in=1)
    assert np.isclose(w.sum(), 1.0)
    assert off.shape[0] == 7
    # through-plane sigma matches FWHM = thickness
    assert np.isclose(2.0 * FWHM_TO_SIGMA, 2.0 / 2.3548200450309493)


def test_psf_delta_mode():
    off, w = gaussian_psf_local(2.0, (1.0, 1.0), mode="delta")
    assert off.shape == (1, 3) and np.allclose(off, 0) and np.isclose(w[0], 1.0)


def test_simulation_shapes_and_affines():
    vol = _phantom(24)
    cfg = {"in_plane_mm": 1.0, "thickness_mm": 2.0, "snr": 0.0,
           "psf": {"mode": "gaussian", "n_through": 5, "n_in": 1},
           "motion": {"enabled": False},
           "stacks": [{"orientation": "axial"}, {"orientation": "coronal"},
                      {"orientation": "sagittal"}]}
    stacks = simulate(vol, cfg, seed=0)
    assert len(stacks) == 3
    for st in stacks:
        assert np.isclose(st.thickness, 2.0)
        assert st.data.max() > 0


def test_load_stacks_dir_excludes_gt(tmp_path=None):
    """Regression: gt.nii.gz / recon*.nii.gz must NOT be loaded as input stacks."""
    import os, tempfile
    from msinr.common import io as mio
    d = tempfile.mkdtemp()
    vol = _phantom(16)
    mio.save_volume(vol, os.path.join(d, "gt.nii.gz"))
    mio.save_volume(vol, os.path.join(d, "recon.nii.gz"))
    from msinr.data.simulate import simulate
    cfg = {"in_plane_mm": 1.0, "thickness_mm": 2.0, "snr": 0.0,
           "psf": {"mode": "gaussian", "n_through": 3, "n_in": 1},
           "motion": {"enabled": False}, "stacks": [{"orientation": "axial"}]}
    for i, st in enumerate(simulate(vol, cfg, seed=0)):
        mio.save_stack(st, os.path.join(d, f"stack_{i:02d}_{st.name}.nii.gz"))
    names = [s.name for s in mio.load_stacks_dir(d)]
    assert "gt" not in names and not any(n.startswith("recon") for n in names), names
    assert len(names) == 1


def test_intensity_matching():
    """A scaled+shifted GT must score ~perfect after affine alignment, poor without."""
    from msinr.common.metrics import all_metrics
    rng = np.random.default_rng(0)
    gt = rng.uniform(0, 1000, (24, 24, 24)).astype(np.float32)
    mask = np.ones(gt.shape, bool)
    pred = 0.3 * gt + 50.0                       # arbitrary intensity scale (like NeSVoR)
    m_aff = all_metrics(pred, gt, mask, match="affine")
    m_none = all_metrics(pred, gt, mask, match="none")
    assert m_aff["psnr"] > 60 and m_aff["psnr"] > m_none["psnr"] + 20, (m_aff, m_none)
    # NCC is scale-invariant -> ~1 regardless of matching
    assert m_aff["ncc"] > 0.999 and abs(m_aff["ncc"] - m_none["ncc"]) < 1e-6


def test_classical_operator_matches_simulation():
    """A_k applied to the GT grid should approximate the simulated observations."""
    vol = _phantom(24)
    cfg = {"in_plane_mm": 1.0, "thickness_mm": 2.0, "snr": 0.0,
           "psf": {"mode": "gaussian", "n_through": 5, "n_in": 1},
           "motion": {"enabled": False},
           "stacks": [{"orientation": "axial"}]}
    stacks = simulate(vol, cfg, seed=0)
    grid = GridSpec.from_volume(vol)
    A, y = build_operator(stacks, grid, foreground_only=False, psf_override=cfg["psf"])
    pred = A @ vol.data.ravel().astype(np.float64)
    # same operator + same field -> tight match (interp differences only)
    assert np.sqrt(np.mean((pred - y) ** 2)) < 0.05
