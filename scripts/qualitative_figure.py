#!/usr/bin/env python
"""No-GT qualitative panel for real reconstructions: three orthogonal centre slices
per method, side by side (plus one input thick-slice stack for reference). Used for
the real Nigerian figures where there is no isotropic ground truth.

    python scripts/qualitative_figure.py --results-dir results/nigerian/sub65 \
        --stacks data/nigerian/sub65 --methods trilinear classical_srr inr_adam inr_muon \
        --out results/nigerian/sub65_qual.png
"""
import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from msinr.common import io as mio


def _orthoviews(vol):
    d = vol.data
    X, Y, Z = d.shape
    return [np.rot90(d[X // 2, :, :]), np.rot90(d[:, Y // 2, :]), d[:, :, Z // 2]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--stacks", default=None, help="Optional: show an input stack slice.")
    ap.add_argument("--methods", nargs="*",
                    default=["trilinear", "classical_srr", "inr_adam", "inr_muon"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cols = []
    if args.stacks:
        stacks = mio.load_stacks_dir(args.stacks)
        if stacks:
            st = stacks[0]
            cols.append((f"input: {st.name}\n(thick {st.thickness:.0f}mm)",
                         [np.rot90(st.data[st.shape[0] // 2, :, :]),
                          np.rot90(st.data[:, st.shape[1] // 2, :]),
                          st.data[:, :, st.shape[2] // 2]]))
    for m in args.methods:
        f = os.path.join(args.results_dir, m, "recon.nii.gz")
        if os.path.exists(f):
            cols.append((m, _orthoviews(mio.load_volume(f))))

    if not cols:
        raise SystemExit("no reconstructions found")
    ncol = len(cols)
    fig, axes = plt.subplots(3, ncol, figsize=(3 * ncol, 9), squeeze=False)
    for c, (title, views) in enumerate(cols):
        for r in range(3):
            ax = axes[r, c]
            img = views[r]
            vmax = np.percentile(img[img > 0], 99) if np.any(img > 0) else 1.0
            ax.imshow(img, cmap="gray", vmin=0, vmax=vmax)
            ax.axis("off")
            if r == 0:
                ax.set_title(title, fontsize=11)
    for r, lbl in enumerate(["sagittal", "coronal", "axial"]):
        axes[r, 0].set_ylabel(lbl, fontsize=10, rotation=90)
    plt.suptitle(f"Reconstruction — {os.path.basename(args.results_dir)}", fontsize=13)
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    plt.savefig(args.out, dpi=140, bbox_inches="tight")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
