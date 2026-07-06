#!/usr/bin/env python
"""Qualitative comparison figure: a slice of GT vs each method's reconstruction,
with error maps. Also shows a low-res input stack slice for reference.

    python scripts/compare_figure.py --subject-dir data/brats/BraTS2021_00000 \
        --results-dir results/brats/BraTS2021_00000 --out results/brats/compare_00000.png
"""
import argparse
import glob
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from msinr.common import io as mio
from msinr.common.metrics import psnr, ssim
from msinr.common.contracts import GridSpec
from msinr.common.resample import resample_to_grid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject-dir", required=True)
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--methods", nargs="*",
                    default=["classical_srr", "inr_adam", "inr_muon"])
    ap.add_argument("--axis", type=int, default=2)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    gt = mio.load_volume(os.path.join(args.subject_dir, "gt.nii.gz"))
    mask = mio.brain_mask(gt)
    sl = gt.shape[args.axis] // 2

    def take(v):
        return np.take(v, sl, axis=args.axis)

    recons = {}
    for m in args.methods:
        f = os.path.join(args.results_dir, m, "recon.nii.gz")
        if os.path.exists(f):
            r = mio.load_volume(f)
            if r.shape != gt.shape:
                r = resample_to_grid(r, GridSpec.from_volume(gt))
            recons[m] = r.data

    ncol = 1 + len(recons)
    fig, axes = plt.subplots(2, ncol, figsize=(3.2 * ncol, 6.4))
    vmax = np.percentile(gt.data[mask], 99)

    axes[0, 0].imshow(np.rot90(take(gt.data)), cmap="gray", vmin=0, vmax=vmax)
    axes[0, 0].set_title("Ground truth", fontsize=11); axes[0, 0].axis("off")
    axes[1, 0].axis("off")

    for i, (m, r) in enumerate(recons.items(), start=1):
        ps = psnr(r, gt.data, mask); ss = ssim(r, gt.data, mask)
        axes[0, i].imshow(np.rot90(take(r)), cmap="gray", vmin=0, vmax=vmax)
        axes[0, i].set_title(f"{m}\nPSNR {ps:.2f}  SSIM {ss:.3f}", fontsize=11)
        axes[0, i].axis("off")
        err = np.abs(take(r) - take(gt.data))
        axes[1, i].imshow(np.rot90(err), cmap="inferno", vmin=0, vmax=vmax * 0.5)
        axes[1, i].set_title("|error|", fontsize=10); axes[1, i].axis("off")

    plt.suptitle(f"Multi-stack SRR — {os.path.basename(args.subject_dir)} "
                 f"(slice {sl}, axis {args.axis})", fontsize=12)
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    plt.savefig(args.out, dpi=140, bbox_inches="tight")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
