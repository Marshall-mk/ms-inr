#!/usr/bin/env python
"""BASELINE: trilinear-averaging reconstruction (registration-free lower bound).

    python methods/trilinear/run.py --stacks data/sub01 --gt data/sub01/gt.nii.gz \
        --out results/sub01/trilinear
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import _bootstrap  # noqa: F401

from msinr import runner
from msinr.trilinear import reconstruct_trilinear


def main():
    ap = runner.base_argparser("Trilinear-averaging baseline.")
    args = ap.parse_args()
    cfg = runner.load_config(args)
    stacks, gt = runner.load_inputs(args)
    result = reconstruct_trilinear(stacks, gt, cfg)
    runner.finalize(result, gt, args.out)


if __name__ == "__main__":
    main()
