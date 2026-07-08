"""Shared plumbing for the isolated ``methods/<name>/run.py`` entry points.

Provides the common CLI (``--stacks --gt --out --config --seed --device`` plus
``--set key=value`` overrides), input loading, and result finalization (masked
metrics vs GT + writing recon.nii.gz / metrics.json / profile.json). Method
scripts import only this + their own reconstruction code -- never each other.
"""
from __future__ import annotations

import argparse
import os

import yaml

import numpy as np

from .common import io as mio
from .common.contracts import Volume, GridSpec, ReconResult
from .common.metrics import all_metrics
from .common.resample import resample_to_grid


def base_argparser(description: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--stacks", required=True, help="Directory of stack NIfTIs.")
    ap.add_argument("--gt", default=None, help="Ground-truth HR NIfTI (optional).")
    ap.add_argument("--out", required=True, help="Output directory.")
    ap.add_argument("--config", default=None, help="YAML config.")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--set", nargs="*", default=[], metavar="k=v",
                    help="Override config entries, e.g. --set iters=500 model=siren_mlp")
    return ap


def _coerce(v: str):
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    if v.lower() in ("true", "false"):
        return v.lower() == "true"
    return v


def load_config(args) -> dict:
    cfg = {}
    if args.config and os.path.exists(args.config):
        with open(args.config) as f:
            cfg = yaml.safe_load(f) or {}
    for item in args.set:
        k, _, v = item.partition("=")
        cfg[k] = _coerce(v)
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.device is not None:
        cfg["device"] = args.device
    return cfg


def load_inputs(args):
    stacks = mio.load_stacks_dir(args.stacks)
    if not stacks:
        raise FileNotFoundError(f"No .nii/.nii.gz stacks found in {args.stacks}")
    gt = mio.load_volume(args.gt, name="gt") if args.gt else None
    return stacks, gt


def finalize(result: ReconResult, gt: Volume | None, out: str,
             recon_name: str = "recon.nii.gz") -> ReconResult:
    """Compute masked metrics vs GT (if any) and write all output artifacts."""
    os.makedirs(out, exist_ok=True)
    if gt is not None:
        mask = mio.brain_mask(gt)
        # a ROI-cropped recon has a different grid than GT -> resample onto GT for metrics
        recon = result.volume
        if recon.shape != gt.shape or not np.allclose(recon.affine, gt.affine):
            recon = resample_to_grid(recon, GridSpec.from_volume(gt))
        result.metrics = all_metrics(recon.data, gt.data, mask,
                                     match=result.config.get("match_intensity", "affine"))
    mio.save_volume(result.volume, os.path.join(out, recon_name))
    result.save_sidecars(os.path.join(out, "metrics.json"),
                         os.path.join(out, "profile.json"))
    _print_summary(result)
    return result


def _print_summary(result: ReconResult):
    m, sec = result.metrics, result.profile.get("sections", {})
    line = f"[{result.method}]"
    if m:
        line += (f" PSNR={m.get('psnr', float('nan')):.2f} "
                 f"SSIM={m.get('ssim', float('nan')):.4f} "
                 f"NRMSE={m.get('nrmse', float('nan')):.4f} "
                 f"NCC={m.get('ncc', float('nan')):.4f}")
    if "reconstruct" in sec:
        line += f" | recon={sec['reconstruct']['seconds']:.1f}s"
    if "inference" in sec:
        line += f" infer={sec['inference']['seconds']:.2f}s"
    if "num_parameters" in result.profile:
        line += f" params={result.profile['num_parameters']:,}"
    print(line)
