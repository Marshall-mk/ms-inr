#!/usr/bin/env python
"""Run the full benchmark: every method x every subject, each in isolation.

Each method is launched as its own subprocess (optionally in its own conda env),
coupled only through the file-based input/output contracts. Failures are recorded
and do not abort the sweep. After running, results are aggregated to a CSV +
figures via ``aggregate.py``.

    python benchmark/run_benchmark.py --config configs/experiment/brats.yaml
"""
import argparse
import os
import subprocess
import sys
import time

import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def build_cmd(method: dict, subject: dict, out: str, env: str):
    cmd = ["conda", "run", "--no-capture-output", "-n", method.get("env", env),
           "python", os.path.join(ROOT, method["run"]),
           "--stacks", subject["stacks"], "--out", out]
    if subject.get("gt"):
        cmd += ["--gt", subject["gt"]]
    if method.get("config"):
        cmd += ["--config", method["config"]]
    sets = method.get("set") or {}
    if sets:
        cmd += ["--set", *[f"{k}={v}" for k, v in sets.items()]]
    return cmd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment/brats.yaml")
    ap.add_argument("--only", nargs="*", help="Restrict to these method names.")
    ap.add_argument("--skip-aggregate", action="store_true")
    ap.add_argument("--subjects-root", default=None,
                    help="Auto-discover subjects: every <dir>/*/ with gt.nii.gz + stacks.")
    ap.add_argument("--n", type=int, default=None, help="Limit auto-discovered subjects.")
    ap.add_argument("--results-dir", default=None, help="Override results_dir from config.")
    args = ap.parse_args()

    with open(args.config) as f:
        bench = yaml.safe_load(f)
    env = bench.get("env", "dev")
    results_dir = args.results_dir or bench["results_dir"]
    os.makedirs(results_dir, exist_ok=True)

    methods = bench["methods"]
    if args.only:
        methods = [m for m in methods if m["name"] in args.only]

    if args.subjects_root:
        import glob
        subs = sorted(d for d in glob.glob(os.path.join(args.subjects_root, "*"))
                      if os.path.exists(os.path.join(d, "gt.nii.gz"))
                      and glob.glob(os.path.join(d, "stack_*.nii.gz")))
        if args.n:
            subs = subs[: args.n]
        bench["subjects"] = [{"name": os.path.basename(d),
                              "stacks": d, "gt": os.path.join(d, "gt.nii.gz")}
                             for d in subs]
        print(f"Auto-discovered {len(bench['subjects'])} subjects under {args.subjects_root}")

    log = []
    for subject in bench["subjects"]:
        for method in methods:
            out = os.path.join(results_dir, subject["name"], method["name"])
            os.makedirs(out, exist_ok=True)
            cmd = build_cmd(method, subject, out, env)
            print(f"\n=== {subject['name']} / {method['name']} ===")
            print(" ", " ".join(cmd))
            t0 = time.perf_counter()
            rc = subprocess.run(cmd).returncode
            dt = time.perf_counter() - t0
            status = "ok" if rc == 0 else f"FAILED(rc={rc})"
            print(f"  -> {status} in {dt:.1f}s")
            log.append({"subject": subject["name"], "method": method["name"],
                        "status": status, "wall_s": round(dt, 1)})

    print("\n=== run summary ===")
    for r in log:
        print(f"  {r['subject']:12s} {r['method']:16s} {r['status']:14s} {r['wall_s']}s")

    if not args.skip_aggregate:
        agg = os.path.join(ROOT, "benchmark", "aggregate.py")
        subprocess.run(["conda", "run", "--no-capture-output", "-n", env,
                        "python", agg, "--results", results_dir])


if __name__ == "__main__":
    main()
