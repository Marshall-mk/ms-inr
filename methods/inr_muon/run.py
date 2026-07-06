#!/usr/bin/env python
"""PROPOSED METHOD: multi-stack SRR with a rank-optimized (Muon) INR.

Runs in isolation:
    python methods/inr_muon/run.py --stacks data/sub01 --gt data/sub01/gt.nii.gz \
        --out results/sub01/inr_muon --config configs/default.yaml

Requires the Muon optimizer:
    pip install "git+https://github.com/KellerJordan/Muon"
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import _bootstrap  # noqa: F401  (adds project root to sys.path)

from msinr import runner
from msinr.recon import reconstruct_inr


def main():
    ap = runner.base_argparser("Rank-optimized (Muon) INR multi-stack SRR.")
    args = ap.parse_args()
    cfg = runner.load_config(args)
    stacks, gt = runner.load_inputs(args)
    result = reconstruct_inr(stacks, gt, cfg, optimizer="muon")
    runner.finalize(result, gt, args.out)


if __name__ == "__main__":
    main()
