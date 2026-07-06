#!/usr/bin/env python
"""Organize the Nigerian GRAND_TEST real anisotropic scans into a standard
multi-stack layout (one contrast, multiple orientations) for real-data SRR.

    python scripts/prep_nigerian.py --src /home/hamza/data/GRAND_TEST/nigerian \
        --contrast T2 --out data/nigerian

For each subject we copy the axial/coronal/sagittal stacks of the chosen contrast
(no contrast agent) into data/nigerian/<sub>/<orientation>.nii.gz and write a Stack
sidecar recording orientation, field strength, and slice_axis. Subjects with fewer
than --min-orient orientations are skipped. These are REAL thick-slice acquisitions
with no isotropic ground truth (reconstruction is evaluated qualitatively / by
leave-one-stack-out).
"""
import argparse
import glob
import json
import os
import re
import shutil

import nibabel as nib
import numpy as np

ORIENTS = ["axial", "coronal", "sagittal"]


def pick_file(subject_dir, orientation, contrast):
    """Pick one file for (orientation, contrast, noCE), preferring run1/no-run."""
    pats = glob.glob(os.path.join(subject_dir, f"{orientation}_{contrast}_noCE_*.nii.gz"))
    pats = [p for p in pats if "CE" not in os.path.basename(p).replace("noCE", "")]
    if not pats:
        return None
    pats.sort(key=lambda p: ("run2" in p, "run" not in p is False, p))
    return pats[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--contrast", default="T2")
    ap.add_argument("--out", default="data/nigerian")
    ap.add_argument("--min-orient", type=int, default=2)
    args = ap.parse_args()

    subjects = sorted(glob.glob(os.path.join(args.src, "sub*")))
    os.makedirs(args.out, exist_ok=True)
    manifest = []
    for sdir in subjects:
        sub = os.path.basename(sdir)
        chosen = {o: pick_file(sdir, o, args.contrast) for o in ORIENTS}
        chosen = {o: f for o, f in chosen.items() if f}
        if len(chosen) < args.min_orient:
            print(f"  skip {sub}: only {len(chosen)} orientation(s)"); continue
        dst_dir = os.path.join(args.out, sub)
        os.makedirs(dst_dir, exist_ok=True)
        for orient, src in chosen.items():
            dst = os.path.join(dst_dir, f"{orient}.nii.gz")
            shutil.copy2(src, dst)
            im = nib.load(src)
            spacing = np.round(im.header.get_zooms()[:3], 3)
            slice_axis = int(np.argmax(spacing))
            fstr = "1.5T" if "1.5T" in src else ("0.3T" if "0.3T" in src else "?")
            with open(dst.replace(".nii.gz", ".json"), "w") as f:
                json.dump({"name": orient, "slice_axis": slice_axis,
                           "meta": {"orientation": orient, "field_strength": fstr,
                                    "source": os.path.basename(src),
                                    "spacing": spacing.tolist()}}, f, indent=2)
        manifest.append(f"{sub}\t{','.join(sorted(chosen))}")
        print(f"  {sub}: {sorted(chosen)}")
    with open(os.path.join(args.out, "subjects.txt"), "w") as f:
        f.write("\n".join(manifest) + "\n")
    print(f"Prepared {len(manifest)} Nigerian subjects -> {args.out}")


if __name__ == "__main__":
    main()
