#!/usr/bin/env python
"""Collate per-run metrics.json + profile.json into a CSV and summary figures.

    python benchmark/aggregate.py --results results/benchmark
Outputs: results.csv, summary.csv, and figures/ (PSNR bars, quality-vs-compute
scatter, and Muon-vs-Adam stable-rank evolution).
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def collect(results_dir: str) -> pd.DataFrame:
    rows = []
    for subject in sorted(os.listdir(results_dir)):
        sdir = os.path.join(results_dir, subject)
        if not os.path.isdir(sdir):
            continue
        for method in sorted(os.listdir(sdir)):
            mdir = os.path.join(sdir, method)
            if not os.path.isdir(mdir):
                continue
            metrics = _load(os.path.join(mdir, "metrics.json"))
            profile = _load(os.path.join(mdir, "profile.json"))
            sec = profile.get("sections", {})
            rows.append({
                "subject": subject, "method": method,
                "status": metrics.get("status", "ok"),
                "psnr": metrics.get("psnr"), "ssim": metrics.get("ssim"),
                "nrmse": metrics.get("nrmse"), "ncc": metrics.get("ncc"),
                "recon_s": sec.get("reconstruct", {}).get("seconds"),
                "infer_s": sec.get("inference", {}).get("seconds"),
                "peak_gpu_mem_mb": sec.get("reconstruct", {}).get("peak_gpu_mem_mb"),
                "energy_j": sec.get("reconstruct", {}).get("energy_j"),
                "num_parameters": profile.get("num_parameters"),
                "device": profile.get("device"),
            })
    return pd.DataFrame(rows)


def make_figures(df: pd.DataFrame, results_dir: str):
    figdir = os.path.join(results_dir, "figures")
    os.makedirs(figdir, exist_ok=True)
    ok = df[df["status"] == "ok"].dropna(subset=["psnr"])
    if not ok.empty:
        agg = ok.groupby("method")[["psnr", "ssim", "nrmse", "ncc"]].mean()
        # PSNR bars
        plt.figure(figsize=(7, 4))
        agg["psnr"].sort_values().plot.bar(color="steelblue")
        plt.ylabel("PSNR (dB)"); plt.title("Reconstruction PSNR by method")
        plt.tight_layout(); plt.savefig(os.path.join(figdir, "psnr_by_method.png"), dpi=130)
        plt.close()
        # quality vs compute
        plt.figure(figsize=(7, 5))
        for method, g in ok.groupby("method"):
            plt.scatter(g["recon_s"], g["psnr"], label=method, s=60)
        plt.xlabel("reconstruction time (s)"); plt.ylabel("PSNR (dB)")
        plt.title("Quality vs compute"); plt.legend(); plt.grid(alpha=0.3)
        plt.tight_layout(); plt.savefig(os.path.join(figdir, "quality_vs_compute.png"), dpi=130)
        plt.close()
    _stable_rank_figure(results_dir, figdir)


def _stable_rank_figure(results_dir: str, figdir: str):
    """Overlay stable-rank evolution for inr_muon vs inr_adam (first subject found)."""
    series = {}
    for subject in sorted(os.listdir(results_dir)):
        for method in ("inr_muon", "inr_adam"):
            prof = _load(os.path.join(results_dir, subject, method, "profile.json"))
            hist = prof.get("history", [])
            if hist and "stable_rank" in hist[0]:
                its = [h["iter"] for h in hist]
                # mean stable rank across hidden layers (drop last 1-D layer)
                sr = [np.mean(h["stable_rank"][:-1]) for h in hist]
                series[f"{method}"] = (its, sr)
        if series:
            break
    if not series:
        return
    plt.figure(figsize=(7, 4))
    for name, (its, sr) in series.items():
        plt.plot(its, sr, marker="o", label=name)
    plt.xlabel("iteration"); plt.ylabel("mean hidden stable rank")
    plt.title("Stable-rank evolution: Muon vs Adam"); plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(figdir, "stable_rank_muon_vs_adam.png"), dpi=130)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/benchmark")
    args = ap.parse_args()

    df = collect(args.results)
    if df.empty:
        print("No results found."); return
    csv = os.path.join(args.results, "results.csv")
    df.to_csv(csv, index=False)

    summary = (df[df["status"] == "ok"]
               .groupby("method")[["psnr", "ssim", "nrmse", "ncc",
                                   "recon_s", "infer_s", "peak_gpu_mem_mb",
                                   "num_parameters"]].mean(numeric_only=True))
    summary.to_csv(os.path.join(args.results, "summary.csv"))
    make_figures(df, args.results)

    pd.set_option("display.width", 160, "display.max_columns", 20)
    print("\n=== per-run results ===")
    print(df.to_string(index=False))
    print("\n=== summary (mean over subjects) ===")
    print(summary.round(3).to_string())
    print(f"\nWrote {csv}, summary.csv, and figures/ under {args.results}")


if __name__ == "__main__":
    main()
