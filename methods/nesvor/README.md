# `nesvor` — optional SOTA INR baseline

Adapter around the official [NeSVoR](https://github.com/daviddmc/NeSVoR) CLI
(Xu et al., IEEE TMI 2023). Defensive: if the `nesvor` command is not on PATH,
the run is recorded as **skipped** and the benchmark continues.

## Install on this box (GB10 / ARM64 / Blackwell)
The prebuilt Docker image (`junshenxu/nesvor`, x86_64 + CUDA 11.7) does **not**
work here. Build from source:
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
