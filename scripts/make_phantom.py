#!/usr/bin/env python
"""Write a small synthetic isotropic brain-like phantom for smoke tests.

Not anatomically meaningful -- just structured intensities (nested ellipsoids +
a few spheres + a sinusoidal texture) so SRR has high-frequency content to
recover. Default 48^3 @ 1mm keeps GPU/CPU cost tiny.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from msinr.common.contracts import Volume
from msinr.common import io as mio


def phantom(n=48):
    lin = np.linspace(-1, 1, n)
    x, y, z = np.meshgrid(lin, lin, lin, indexing="ij")
    r2 = x**2 + y**2 + z**2
    vol = np.zeros((n, n, n), np.float32)
    vol[r2 < 0.85**2] = 0.4                       # skull-ish shell
    vol[r2 < 0.75**2] = 0.9                       # brain
    vol[(x**2 / 0.5**2 + y**2 / 0.3**2 + z**2 / 0.4**2) < 1] = 0.6   # ventricle-ish
    for c, rad, val in [((-0.3, 0.2, 0.1), 0.12, 1.0),
                        ((0.35, -0.15, -0.2), 0.10, 0.2),
                        ((0.0, 0.4, -0.3), 0.08, 1.2)]:
        vol[((x - c[0])**2 + (y - c[1])**2 + (z - c[2])**2) < rad**2] = val
    texture = 0.08 * np.sin(12 * np.pi * x) * np.cos(12 * np.pi * y) * np.sin(12 * np.pi * z)
    vol[r2 < 0.75**2] += texture[r2 < 0.75**2]
    return np.clip(vol, 0, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--spacing", type=float, default=1.0)
    ap.add_argument("--out", default="data/phantom/gt.nii.gz")
    args = ap.parse_args()
    data = phantom(args.n)
    affine = np.eye(4)
    affine[0, 0] = affine[1, 1] = affine[2, 2] = args.spacing
    affine[:3, 3] = -0.5 * args.spacing * (args.n - 1)   # centre at origin
    mio.save_volume(Volume(data, affine, name="gt"), args.out)
    print(f"Wrote phantom {data.shape} spacing={args.spacing}mm -> {args.out} "
          f"(range {data.min():.2f}..{data.max():.2f})")


if __name__ == "__main__":
    main()
