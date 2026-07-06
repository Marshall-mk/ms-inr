# `inr_muon` — proposed method

Multi-stack super-resolution reconstruction with a coordinate INR trained via the
**Muon optimizer** (high-rank, near-orthogonal updates; McGinnis et al. 2025,
arXiv:2512.14366). Muon optimizes the hidden weight matrices; first/last layers
and biases use an auxiliary Adam.

## Install
Uses the `dev` conda env plus Muon:
```bash
pip install "git+https://github.com/KellerJordan/Muon"
```

## Run (isolated)
```bash
conda run -n dev python methods/inr_muon/run.py \
  --stacks data/sub01 --gt data/sub01/gt.nii.gz \
  --out results/sub01/inr_muon --config configs/default.yaml \
  --set iters=3000 muon_lr=0.01 lr=0.001
```

## Contract
- **Input:** `--stacks` dir of NIfTI stacks (+ json sidecars from simulation) and
  optional `--gt` HR NIfTI.
- **Output:** `recon.nii.gz`, `metrics.json` (PSNR/SSIM/NRMSE/NCC, brain-masked),
  `profile.json` (time, peak GPU mem, params, throughput, stable-rank history).

Key config keys: `model, hidden_dim, num_layers, sigma, iters, muon_lr, lr,
batch_per_stack`. See `configs/default.yaml`.
