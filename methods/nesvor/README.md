# `nesvor` — optional SOTA INR baseline

Adapter around the official [NeSVoR](https://github.com/daviddmc/NeSVoR) CLI
(Xu et al., IEEE TMI 2023). Defensive: if the `nesvor` command is not on PATH,
the run is recorded as **skipped** and the benchmark continues.

## On the HPC server (x86_64 + A30) — recommended
The A30 nodes are x86_64/Ampere, so the **official Docker image works** via
Singularity/Apptainer (no root needed). Just run the orchestration script from the
repo root — it pulls `docker://junshenxu/nesvor`, wraps it as a `nesvor` CLI on PATH,
runs this adapter per subject into `results/africai/brats_5mm`, and re-aggregates:
```bash
sbatch run_nesvor.sh          # pre-pull on a login node if compute nodes are offline
```

## On the GB10 dev box (ARM64 / Blackwell) — from source
The prebuilt Docker image (x86_64 + CUDA 11.7) does **not** work on ARM/Blackwell.
Build from source:
1. Ensure the `dev` env has aarch64 PyTorch cu12.8+ (already present: torch 2.12 / cu13).
2. Build tiny-cuda-nn from master with the Blackwell arch:
   ```bash
   TCNN_CUDA_ARCHITECTURES=120 \
     pip install "git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch"
   ```
   If the build fails with a C++14/C++17 error (tcnn issue #527), force C++17 in
   the extension's nvcc/host flags and retry.
3. `pip install git+https://github.com/daviddmc/NeSVoR.git`

## Run (isolated)
```bash
conda run -n dev python methods/nesvor/run.py \
  --stacks data/sub01 --gt data/sub01/gt.nii.gz \
  --out results/sub01/nesvor --set output_resolution=1.0
```

The adapter resamples NeSVoR's output onto the GT grid before computing metrics.
