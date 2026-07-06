#!/usr/bin/env python
"""BASELINE: classical least-squares SRR (Tikhonov-regularized, CG-solved),
using the same PSF forward model as the INR methods. Learning-free.

    python methods/classical_srr/run.py --stacks data/sub01 --gt data/sub01/gt.nii.gz \
        --out results/sub01/classical_srr --set reg_lambda=0.1 cg_maxiter=200
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import _bootstrap  # noqa: F401

from msinr import runner
from msinr.classical import reconstruct_classical


def main():
    ap = runner.base_argparser("Classical LS-SRR (CG) multi-stack reconstruction.")
    args = ap.parse_args()
    cfg = runner.load_config(args)
    stacks, gt = runner.load_inputs(args)
    result = reconstruct_classical(stacks, gt, cfg)
    runner.finalize(result, gt, args.out)


if __name__ == "__main__":
    main()
