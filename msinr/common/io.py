"""NIfTI I/O and a simple brain mask, on top of nibabel."""
from __future__ import annotations

import json
import os

import numpy as np
import nibabel as nib
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu

from .contracts import Volume, Stack


# ---------------------------------------------------------------------------
# Volumes
# ---------------------------------------------------------------------------
def load_volume(path: str, name: str | None = None) -> Volume:
    img = nib.load(path)
    data = np.asanyarray(img.dataobj, dtype=np.float32)
    if data.ndim > 3:
        data = data[..., 0]           # drop trailing singleton/time dims
    return Volume(data=data, affine=img.affine.astype(np.float64),
                  name=name or _stem(path))


def save_volume(vol: Volume, path: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    nib.save(nib.Nifti1Image(vol.data.astype(np.float32), vol.affine), path)


# ---------------------------------------------------------------------------
# Stacks (NIfTI + json sidecar for metadata)
# ---------------------------------------------------------------------------
def save_stack(stack: Stack, path: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    nib.save(nib.Nifti1Image(stack.data.astype(np.float32), stack.affine), path)
    sidecar = _sidecar_path(path)
    with open(sidecar, "w") as f:
        json.dump({"name": stack.name, "slice_axis": stack.slice_axis,
                   "meta": _jsonable(stack.meta)}, f, indent=2)


def load_stack(path: str, name: str | None = None) -> Stack:
    img = nib.load(path)
    data = np.asanyarray(img.dataobj, dtype=np.float32)
    if data.ndim > 3:
        data = data[..., 0]
    slice_axis, meta = 2, {}
    sidecar = _sidecar_path(path)
    if os.path.exists(sidecar):
        with open(sidecar) as f:
            d = json.load(f)
        slice_axis = d.get("slice_axis", 2)
        meta = d.get("meta", {})
        name = name or d.get("name")
    return Stack(data=data, affine=img.affine.astype(np.float64),
                 name=name or _stem(path), slice_axis=slice_axis, meta=meta)


def load_stacks_dir(stacks_dir: str) -> list[Stack]:
    """Load every .nii/.nii.gz in a directory as a Stack (sorted by name)."""
    files = [f for f in os.listdir(stacks_dir)
             if f.endswith((".nii", ".nii.gz"))]
    files.sort()
    return [load_stack(os.path.join(stacks_dir, f)) for f in files]


# ---------------------------------------------------------------------------
# Brain mask (intensity + morphology). Good enough for masked metrics;
# swap for a real skull-strip if the user provides one.
# ---------------------------------------------------------------------------
def brain_mask(vol: Volume, closing_radius: int = 2) -> np.ndarray:
    data = vol.data
    pos = data[data > 0]
    if pos.size == 0:
        return np.ones(data.shape, dtype=bool)
    try:
        thr = threshold_otsu(pos)
    except ValueError:
        thr = pos.mean()
    mask = data > thr
    # keep the largest connected component, then close + fill holes
    lbl, n = ndi.label(mask)
    if n > 1:
        sizes = ndi.sum(np.ones_like(lbl), lbl, index=range(1, n + 1))
        mask = lbl == (int(np.argmax(sizes)) + 1)
    if closing_radius > 0:
        st = _ball(closing_radius)
        mask = ndi.binary_closing(mask, structure=st)
    mask = ndi.binary_fill_holes(mask)
    return mask.astype(bool)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _ball(radius: int) -> np.ndarray:
    r = radius
    zz, yy, xx = np.ogrid[-r:r + 1, -r:r + 1, -r:r + 1]
    return (xx * xx + yy * yy + zz * zz) <= r * r


def _stem(path: str) -> str:
    base = os.path.basename(path)
    for ext in (".nii.gz", ".nii"):
        if base.endswith(ext):
            return base[: -len(ext)]
    return os.path.splitext(base)[0]


def _sidecar_path(path: str) -> str:
    return _stem_path(path) + ".json"


def _stem_path(path: str) -> str:
    for ext in (".nii.gz", ".nii"):
        if path.endswith(ext):
            return path[: -len(ext)]
    return os.path.splitext(path)[0]


def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj
