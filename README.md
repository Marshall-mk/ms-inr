# ms-inr — Rank-Optimized INRs for Multi-Stack MRI Super-Resolution

Reproducible benchmark for **multi-stack super-resolution reconstruction (SRR)**
of adult brain MRI. The **proposed method** is a coordinate Implicit Neural
Representation (INR) trained with the **Muon optimizer** — whose high-rank,
near-orthogonal updates raise INR fidelity (McGinnis et al., *Optimizing Rank for
High-Fidelity INRs*, arXiv:2512.14366) — coupled with a 3D multi-stack acquisition
forward model. It is compared against baselines, with **inference time and compute
tracked** for every method.

## Methods (each runs in isolation)
| Method | Dir | What |
| --- | --- | --- |
| **INR + Muon** (proposed) | `methods/inr_muon` | Rank-optimized INR SRR |
| INR + Adam (ablation) | `methods/inr_adam` | Same INR, Adam — isolates Muon's effect |
| Classical LS-SRR | `methods/classical_srr` | CG least-squares, same PSF operator |
| NeSVoR (optional) | `methods/nesvor` | Official INR SVR CLI; auto-skips if absent |

Every method shares one **contract**: input = a dir of stack NIfTIs (+ optional GT
NIfTI); output = `recon.nii.gz` + `metrics.json` (PSNR/SSIM/NRMSE/NCC, brain-masked)
+ `profile.json` (time, peak GPU mem, params, throughput, energy).

## Pipeline
```
                 simulate                 reconstruct                evaluate
isotropic HR  ─────────────►  3 orthogonal  ─────────────►  isotropic  ──────────►  metrics + compute
 volume (GT)   PSF+motion+     thick stacks   method/run.py   recon        vs GT       tables/figures
               downsample      (+ affines)
```

## Quickstart
```bash
# 0) env: use the `dev` conda env (has torch cu13). Install Muon for the proposed method:
pip install "git+https://github.com/KellerJordan/Muon"

# 1) make a tiny synthetic phantom (or point at your own isotropic HR NIfTI)
conda run -n dev python scripts/make_phantom.py --out data/phantom/gt.nii.gz

# 2) simulate 3 orthogonal thick-slice stacks
conda run -n dev python -m msinr.data.simulate \
  --input data/phantom/gt.nii.gz --out data/phantom \
  --config configs/simulate/default.yaml

# 3) run one method in isolation (phantom is tiny -> shrink the model via --set)
conda run -n dev python methods/inr_muon/run.py \
  --stacks data/phantom --gt data/phantom/gt.nii.gz \
  --out results/phantom/inr_muon --config configs/default.yaml \
  --set hidden_dim=256 num_layers=4 sigma=3 mapping_size=128 iters=800

# 4) real benchmark: prep BraTS, simulate stacks, run all methods + aggregate
conda run -n dev python scripts/prep_brats.py --src <BRATS_DIR> --n 50 --out data/brats
conda run -n dev python scripts/simulate_batch.py --root data/brats --config configs/simulate/default.yaml
conda run -n dev python benchmark/run_benchmark.py --config configs/experiment/brats.yaml
```

## Using your own data
Point `--input` (simulation) at any isotropic HR NIfTI, or add a subject to a
`configs/experiment/*.yaml` with its `stacks` dir and `gt`. Real thick-slice
acquisitions (no GT, e.g. `configs/experiment/nigerian.yaml`) work too — metrics
are simply omitted.

## Layout
```
msinr/       core library (common/, data/, forward/, models/, recon.py, classical.py)
methods/     isolated per-method entry points (run.py + README + requirements)
benchmark/   run_benchmark.py + aggregate.py
configs/     default.yaml (reconstruction), simulate/, experiment/ (brats, nigerian)
scripts/     make_phantom, prep_brats, prep_nigerian, simulate_batch, compare_figure
tests/       fast CPU sanity tests
```

## Simulation defaults (NeSVoR adult regime)
3 orthogonal stacks · 1mm in-plane / 2mm thick · anisotropic Gaussian PSF
(through-plane FWHM = thickness) · rigid inter-stack motion ±3mm/±6° · Rician noise.
Override in `configs/simulate/`. A `mode: delta` PSF gives the IREM sampling-only model.

## Environment
Developed on an NVIDIA **GB10** (Grace-Blackwell, ARM64, sm_120). Keep GPU runs
modest — start small and scale up. See `methods/nesvor/README.md` for the
from-source NeSVoR build (the prebuilt Docker image does not run on this arch).
