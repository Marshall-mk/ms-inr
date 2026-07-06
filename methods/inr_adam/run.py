#!/usr/bin/env python
"""ABLATION BASELINE: identical INR + forward model, trained with Adam instead of
Muon. The only difference from the proposed method is ``optimizer="adam"``, which
isolates the effect of Muon's high-rank updates (the paper's core claim).

    python methods/inr_adam/run.py --stacks data/sub01 --gt data/sub01/gt.nii.gz \
        --out results/sub01/inr_adam --config configs/default.yaml
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import _bootstrap  # noqa: F401

from msinr import runner
from msinr.recon import reconstruct_inr


def main():
    ap = runner.base_argparser("Adam-INR multi-stack SRR (ablation).")
    args = ap.parse_args()
    cfg = runner.load_config(args)
    stacks, gt = runner.load_inputs(args)
    result = reconstruct_inr(stacks, gt, cfg, optimizer="adam")
    runner.finalize(result, gt, args.out)


if __name__ == "__main__":
    main()
