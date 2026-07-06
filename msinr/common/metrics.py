"""Reconstruction-quality metrics (NeSVoR protocol: PSNR / SSIM / NRMSE / NCC),
all computed **inside a brain mask** by default.
"""
from __future__ import annotations

import numpy as np
from skimage.metrics import structural_similarity


def _prep(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray | None):
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    if mask is None:
        mask = np.ones(gt.shape, dtype=bool)
    else:
        mask = np.asarray(mask, dtype=bool)
    return pred, gt, mask


def data_range(gt: np.ndarray, mask: np.ndarray | None = None) -> float:
    _, gt, mask = _prep(gt, gt, mask)
    vals = gt[mask]
    if vals.size == 0:
        return 1.0
    dr = float(vals.max() - vals.min())
    return dr if dr > 0 else 1.0


def psnr(pred, gt, mask=None, dr: float | None = None) -> float:
    pred, gt, mask = _prep(pred, gt, mask)
    if dr is None:
        dr = data_range(gt, mask)
    mse = float(np.mean((pred[mask] - gt[mask]) ** 2))
    if mse <= 0:
        return float("inf")
    return float(20 * np.log10(dr) - 10 * np.log10(mse))


def nrmse(pred, gt, mask=None) -> float:
    """RMSE normalized by the GT intensity range over the mask."""
    pred, gt, mask = _prep(pred, gt, mask)
    rmse = float(np.sqrt(np.mean((pred[mask] - gt[mask]) ** 2)))
    dr = data_range(gt, mask)
    return rmse / dr


def ncc(pred, gt, mask=None) -> float:
    """Normalized cross-correlation (zero-mean, unit-variance) over the mask."""
    pred, gt, mask = _prep(pred, gt, mask)
    a, b = pred[mask], gt[mask]
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a * a).sum() * (b * b).sum())
    if denom <= 0:
        return 0.0
    return float((a * b).sum() / denom)


def ssim(pred, gt, mask=None, dr: float | None = None) -> float:
    """3D SSIM; if a mask is given, the SSIM map is averaged inside the mask."""
    pred, gt, mask = _prep(pred, gt, mask)
    if dr is None:
        dr = data_range(gt, mask)
    _, ssim_map = structural_similarity(gt, pred, data_range=dr, full=True)
    if mask.all():
        return float(ssim_map.mean())
    return float(ssim_map[mask].mean())


def all_metrics(pred, gt, mask=None) -> dict:
    """All quality metrics as a dict; NaN-safe if inputs mismatch."""
    pred, gt, mask = _prep(pred, gt, mask)
    dr = data_range(gt, mask)
    return {
        "psnr": psnr(pred, gt, mask, dr),
        "ssim": ssim(pred, gt, mask, dr),
        "nrmse": nrmse(pred, gt, mask),
        "ncc": ncc(pred, gt, mask),
        "n_mask_voxels": int(mask.sum()),
        "data_range": dr,
    }
