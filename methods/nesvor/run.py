#!/usr/bin/env python
"""BASELINE (optional): adapter around the official NeSVoR CLI.

NeSVoR (Xu et al., IEEE TMI 2023) is the INR slice-to-volume SOTA. It depends on
tiny-cuda-nn; on this GB10 (ARM64/Blackwell) it needs a from-source build
(aarch64 torch cu128 + tcnn master, TCNN_CUDA_ARCHITECTURES=120, C++17 patch).
The prebuilt Docker image (x86_64/CUDA 11.7) does NOT work here.

This adapter is defensive: if the ``nesvor`` CLI is not on PATH it writes a
"skipped" result and exits 0, so the benchmark proceeds without it.

    python methods/nesvor/run.py --stacks data/sub01 --gt data/sub01/gt.nii.gz \
        --out results/sub01/nesvor --set output_resolution=1.0
"""
import glob
import json
import os
import shutil
import subprocess
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import _bootstrap  # noqa: F401

from msinr import runner
from msinr.common import io as mio
from msinr.common.contracts import Volume, GridSpec, ReconResult
from msinr.common.metrics import all_metrics
from msinr.common.resample import resample_to_grid


def _skip(out, reason):
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "metrics.json"), "w") as f:
        json.dump({"method": "nesvor", "status": "skipped", "reason": reason}, f, indent=2)
    with open(os.path.join(out, "profile.json"), "w") as f:
        json.dump({"method": "nesvor", "status": "skipped", "reason": reason}, f, indent=2)
    print(f"[nesvor] SKIPPED: {reason}")


def main():
    ap = runner.base_argparser("NeSVoR adapter (optional baseline).")
    args = ap.parse_args()
    cfg = runner.load_config(args)
    os.makedirs(args.out, exist_ok=True)

    if shutil.which("nesvor") is None:
        _skip(args.out, "nesvor CLI not found on PATH (not installed on this box).")
        return

    stack_files = sorted(glob.glob(os.path.join(args.stacks, "*.nii.gz"))
                         + glob.glob(os.path.join(args.stacks, "*.nii")))
    stack_files = [f for f in stack_files if os.path.basename(f) != "gt.nii.gz"]
    if not stack_files:
        _skip(args.out, f"no stacks in {args.stacks}")
        return

    out_vol = os.path.join(args.out, "recon.nii.gz")
    res = cfg.get("output_resolution", cfg.get("iso_mm", 1.0))
    cmd = ["nesvor", "reconstruct", "--input-stacks", *stack_files,
           "--output-volume", out_vol, "--output-resolution", str(res)]
    if "nesvor_extra" in cfg:
        cmd += str(cfg["nesvor_extra"]).split()

    print("[nesvor] running:", " ".join(cmd))
    t0 = time.perf_counter()
    try:
        subprocess.run(cmd, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        _skip(args.out, f"nesvor reconstruct failed: {e}")
        return
    dt = time.perf_counter() - t0

    result = ReconResult(volume=mio.load_volume(out_vol, name="recon_nesvor"),
                         method="nesvor", config=dict(cfg),
                         profile={"device": "cuda",
                                  "sections": {"reconstruct": {"seconds": dt},
                                               "inference": {"seconds": 0.0}}})
    gt = mio.load_volume(args.gt, name="gt") if args.gt else None
    if gt is not None:
        # NeSVoR chooses its own grid -> resample onto the GT grid for metrics
        rs = resample_to_grid(result.volume, GridSpec.from_volume(gt))
        result.metrics = all_metrics(rs.data, gt.data, mio.brain_mask(gt))
    result.save_sidecars(os.path.join(args.out, "metrics.json"),
                         os.path.join(args.out, "profile.json"))
    runner._print_summary(result)


if __name__ == "__main__":
    main()
