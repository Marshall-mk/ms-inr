#!/usr/bin/env python
"""Leave-one-stack-out (LOSO) cross-validation for REAL data with no ground truth.

For a subject with K stacks, hold out each stack in turn, reconstruct from the other
K-1, then predict the held-out stack through the PSF forward operator and score it
against the real acquired slices. This yields a rigorous quantitative metric without
any isotropic ground truth -- the core real-data evaluation for the paper.

    python scripts/leave_one_stack_out.py --stacks data/nigerian/sub65 \
        --method inr_muon --config configs/default.yaml \
        --set normalize_stacks=per_stack --out results/loso/sub65_inr_muon

Metrics are computed on the held-out stack after per-image p99 normalization (so the
comparison is scale-invariant across scanners): PSNR/SSIM on foreground voxels.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from msinr.common import io as mio
from msinr.common.contracts import GridSpec
from msinr.common.metrics import psnr as mpsnr, ssim as mssim
from msinr.classical import build_operator, reconstruct_classical
from msinr.trilinear import reconstruct_trilinear


def reconstruct(method, stacks, cfg):
    if method in ("inr_muon", "inr_adam"):
        from msinr.recon import reconstruct_inr
        opt = "muon" if method == "inr_muon" else "adam"
        return reconstruct_inr(stacks, None, cfg, optimizer=opt).volume
    if method == "classical":
        return reconstruct_classical(stacks, None, cfg).volume
    if method == "trilinear":
        return reconstruct_trilinear(stacks, None, cfg).volume
    raise ValueError(f"unknown method {method}")


def _p99_norm(x):
    p = np.percentile(x[x > 0], 99) if np.any(x > 0) else 1.0
    return x / (p + 1e-8)


def main():
    ap = argparse.ArgumentParser(description="Leave-one-stack-out evaluation (no GT).")
    ap.add_argument("--stacks", required=True)
    ap.add_argument("--method", required=True,
                    choices=["inr_muon", "inr_adam", "classical", "trilinear"])
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--out", required=True)
    ap.add_argument("--set", nargs="*", default=[], metavar="k=v")
    args = ap.parse_args()

    import yaml
    cfg = yaml.safe_load(open(args.config)) if os.path.exists(args.config) else {}
    for kv in args.set:
        k, _, v = kv.partition("=")
        for cast in (int, float):
            try:
                v = cast(v); break
            except ValueError:
                pass
        if isinstance(v, str) and v.lower() in ("true", "false"):
            v = v.lower() == "true"
        cfg[k] = v

    stacks = mio.load_stacks_dir(args.stacks)
    if len(stacks) < 2:
        raise SystemExit(f"LOSO needs >=2 stacks, found {len(stacks)} in {args.stacks}")

    os.makedirs(args.out, exist_ok=True)
    folds = []
    for h in range(len(stacks)):
        held = stacks[h]
        train = [s for i, s in enumerate(stacks) if i != h]
        recon = reconstruct(args.method, train, cfg)      # Volume on a grid

        # predict the held-out stack via its PSF forward operator applied to recon
        grid = GridSpec.from_volume(recon)
        A_h, _ = build_operator([held], grid, foreground_only=False,
                                psf_override=cfg.get("psf"))
        pred = (A_h @ recon.data.ravel().astype(np.float64)).reshape(held.shape)
        obs = held.data.astype(np.float64)

        pn, on = _p99_norm(pred), _p99_norm(obs)
        mask = obs > 0
        ps = mpsnr(pn, on, mask); ss = mssim(pn, on, mask)
        folds.append({"held_out": held.name, "psnr": ps, "ssim": ss,
                      "n_fg": int(mask.sum())})
        print(f"  [{args.method}] held-out {held.name:9s} PSNR={ps:.2f} SSIM={ss:.4f}")

    summary = {"method": args.method, "subject": os.path.basename(args.stacks.rstrip("/")),
               "folds": folds,
               "psnr_mean": float(np.mean([f["psnr"] for f in folds])),
               "ssim_mean": float(np.mean([f["ssim"] for f in folds]))}
    with open(os.path.join(args.out, "loso.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[{args.method}] {summary['subject']} LOSO mean "
          f"PSNR={summary['psnr_mean']:.2f} SSIM={summary['ssim_mean']:.4f} -> {args.out}/loso.json")


if __name__ == "__main__":
    main()
