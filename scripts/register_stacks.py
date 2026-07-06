#!/usr/bin/env python
"""Rigidly co-register multi-orientation stacks (real low-field data) before SRR.

Real 0.3T/1.5T acquisitions have inter-scan motion, so the stored affines do not
place the stacks in a perfectly consistent world frame. We rigidly register each
stack to a reference (SimpleITK, Mattes MI, multi-resolution) and **update only the
voxel->world affine** -- no resampling, so the native thick slices are preserved for
the PSF forward model.

    python scripts/register_stacks.py --stacks data/nigerian/sub65 \
        --out data/nigerian_reg/sub65 [--ref axial]

Registration is in SimpleITK/LPS; the affine update is composed in nibabel/RAS
(D = diag(-1,-1,1)) so the rest of the pipeline (nibabel) stays consistent.
"""
import argparse
import glob
import json
import os
import shutil

import numpy as np
import nibabel as nib
import SimpleITK as sitk

D = np.diag([-1.0, -1.0, 1.0, 1.0])   # RAS <-> LPS


def register(fixed, moving):
    """Return the Euler3D transform T mapping fixed-space points to moving-space."""
    init = sitk.CenteredTransformInitializer(
        fixed, moving, sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY)
    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    R.SetMetricSamplingStrategy(R.RANDOM)
    R.SetMetricSamplingPercentage(0.1, seed=42)
    R.SetInterpolator(sitk.sitkLinear)
    R.SetOptimizerAsRegularStepGradientDescent(
        learningRate=1.0, minStep=1e-4, numberOfIterations=200,
        gradientMagnitudeTolerance=1e-6)
    R.SetOptimizerScalesFromPhysicalShift()
    R.SetInitialTransform(init, inPlace=False)
    R.SetShrinkFactorsPerLevel([4, 2, 1])
    R.SetSmoothingSigmasPerLevel([2, 1, 0])
    R.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    return R.Execute(sitk.Cast(fixed, sitk.sitkFloat32),
                     sitk.Cast(moving, sitk.sitkFloat32))


def transform_to_matrix(T):
    """Any SimpleITK transform (rigid) -> 4x4 homogeneous (LPS), by sampling points.

    Works regardless of the concrete transform type (Euler3D / CompositeTransform).
    """
    o = np.array(T.TransformPoint((0.0, 0.0, 0.0)))
    cols = [np.array(T.TransformPoint(tuple(float(x) for x in e))) - o
            for e in np.eye(3)]
    M = np.eye(4)
    M[:3, :3] = np.stack(cols, axis=1)
    M[:3, 3] = o
    return M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stacks", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ref", default=None,
                    help="Reference stack name substring (default: most slices).")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.stacks, "*.nii.gz"))
                   + glob.glob(os.path.join(args.stacks, "*.nii")))
    files = [f for f in files if os.path.basename(f) != "gt.nii.gz"]
    if len(files) < 2:
        raise SystemExit(f"need >=2 stacks in {args.stacks}")

    imgs = {f: sitk.ReadImage(f, sitk.sitkFloat32) for f in files}
    if args.ref:
        ref_file = next(f for f in files if args.ref in os.path.basename(f))
    else:
        ref_file = max(files, key=lambda f: imgs[f].GetNumberOfPixels())
    print(f"reference: {os.path.basename(ref_file)}")

    os.makedirs(args.out, exist_ok=True)
    for f in files:
        nb = nib.load(f)
        A = nb.affine.astype(np.float64)
        base = os.path.basename(f)
        if f == ref_file:
            new_A, note = A, "reference (unchanged)"
        else:
            T = register(imgs[ref_file], imgs[f])
            T_lps = transform_to_matrix(T)
            T_ras = D @ T_lps @ D
            new_A = np.linalg.inv(T_ras) @ A          # re-place moving in ref frame
            shift = np.linalg.norm(new_A[:3, 3] - A[:3, 3])
            note = f"registered (origin shift {shift:.1f}mm)"
        print(f"  {base:32s} {note}")
        nib.save(nib.Nifti1Image(np.asanyarray(nb.dataobj), new_A), os.path.join(args.out, base))
        # carry the sidecar forward
        sc = f.rsplit(".nii", 1)[0] + ".json"
        if os.path.exists(sc):
            with open(sc) as g:
                meta = json.load(g)
            meta.setdefault("meta", {})["registered_to"] = os.path.basename(ref_file)
            with open(os.path.join(args.out, base.rsplit(".nii", 1)[0] + ".json"), "w") as g:
                json.dump(meta, g, indent=2)
    print(f"Wrote registered stacks -> {args.out}")


if __name__ == "__main__":
    main()
