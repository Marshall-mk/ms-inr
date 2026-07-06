#!/usr/bin/env python
"""Copy the first N BraTS subjects' chosen contrast into a standard layout as
isotropic ground-truth volumes for the simulated SRR benchmark.

    python scripts/prep_brats.py --src /home/hamza/data/BraTS2021_Training_Data \
        --n 50 --contrast t2 --out data/brats

Result: data/brats/<ID>/gt.nii.gz for each subject, plus subjects.txt manifest.
BraTS volumes are 1mm isotropic and skull-stripped -- ideal HR ground truth.
"""
import argparse
import os
import shutil


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--contrast", default="t2", choices=["t1", "t1ce", "t2", "flair"])
    ap.add_argument("--out", default="data/brats")
    args = ap.parse_args()

    subjects = sorted(d for d in os.listdir(args.src)
                      if os.path.isdir(os.path.join(args.src, d)))[: args.n]
    os.makedirs(args.out, exist_ok=True)
    manifest = []
    for sid in subjects:
        src = os.path.join(args.src, sid, f"{sid}_{args.contrast}.nii.gz")
        if not os.path.exists(src):
            print(f"  WARN missing {src}"); continue
        dst_dir = os.path.join(args.out, sid)
        os.makedirs(dst_dir, exist_ok=True)
        shutil.copy2(src, os.path.join(dst_dir, "gt.nii.gz"))
        manifest.append(sid)
    with open(os.path.join(args.out, "subjects.txt"), "w") as f:
        f.write("\n".join(manifest) + "\n")
    print(f"Copied {len(manifest)} BraTS {args.contrast} volumes -> {args.out}")


if __name__ == "__main__":
    main()
