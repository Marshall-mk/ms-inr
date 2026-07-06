#!/usr/bin/env python
"""Batch-simulate multi-stack acquisitions for every subject under a root dir that
contains a gt.nii.gz (e.g. data/brats/<ID>/gt.nii.gz).

    python scripts/simulate_batch.py --root data/brats \
        --config configs/simulate/default.yaml [--n 5] [--overwrite]
"""
import argparse
import glob
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from msinr.common import io as mio
from msinr.data.simulate import simulate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--config", default="configs/simulate/default.yaml")
    ap.add_argument("--n", type=int, default=None, help="limit to first N subjects")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--set", nargs="*", default=[], metavar="k=v",
                    help="Override top-level sim keys, e.g. --set thickness_mm=6 snr=10")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    for kv in args.set:
        k, _, v = kv.partition("=")
        try:
            v = float(v) if "." in v else int(v)
        except ValueError:
            pass
        cfg[k] = v
    subj_dirs = sorted(d for d in glob.glob(os.path.join(args.root, "*"))
                       if os.path.exists(os.path.join(d, "gt.nii.gz")))
    if args.n:
        subj_dirs = subj_dirs[: args.n]

    for i, d in enumerate(subj_dirs):
        existing = glob.glob(os.path.join(d, "stack_*.nii.gz"))
        if existing and not args.overwrite:
            print(f"[{i+1}/{len(subj_dirs)}] {os.path.basename(d)}: stacks exist, skip")
            continue
        hr = mio.load_volume(os.path.join(d, "gt.nii.gz"), name="gt")
        stacks = simulate(hr, cfg, seed=args.seed)
        for j, st in enumerate(stacks):
            mio.save_stack(st, os.path.join(d, f"stack_{j:02d}_{st.name}.nii.gz"))
        print(f"[{i+1}/{len(subj_dirs)}] {os.path.basename(d)}: {len(stacks)} stacks "
              f"({', '.join(f'{s.name} {tuple(s.shape)}' for s in stacks)})")


if __name__ == "__main__":
    main()
