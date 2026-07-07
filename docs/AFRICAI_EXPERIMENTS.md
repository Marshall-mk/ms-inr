# AfriCAI 2026 — Experiment Plan

Living plan for the paper *"Training-free super-resolution reconstruction of low-field
African MRI via rank-optimized implicit neural representations."* Each experiment lists
its **goal**, **why it matters for AfriCAI**, the **command(s)** to run, **outputs**, and
any **infrastructure to build first**. Run experiments when the GPU is free (see
`memory`: use the `dev` conda env; start small; GPU is shared).

Target: MICCAI AfriCAI 2026 workshop, LNCS 8+2 pages, double-blind. Verify exact deadline
on africai.org (≈ July 2026 — tight).

---

## 0. Positioning (the sell)

- **Problem is intrinsically low-resource/African:** low-field scanners (our Nigerian
  **0.3T & 1.5T** data) produce thick-slice, anisotropic, motion-prone scans; isotropic
  high-field MRI is scarce. SRR recovers isotropic, diagnostically richer volumes **without
  new hardware**.
- **Headline advantage:** the method is **training-free / scan-specific / self-supervised**
  — no labels, no cohort, no domain-shift. Directly answers Africa's data-scarcity problem.
- **Novelty:** first use of **Muon (rank-optimized) INR training** for slice-to-volume SRR,
  with a mechanistic stable-rank explanation and a clean win over Adam-INR and classical.
- **Validation:** controlled simulated benchmark (BraTS, GT) + real multi-site low-field
  (Nigerian 0.3T/1.5T) + efficiency/deployability.

**One-liner:** *Training-free, scan-specific super-resolution that makes low-field African
MRI isotropic and high-fidelity, using rank-optimized INRs — validated on real 0.3T/1.5T data.*

---

## 1. Datasets

| Name | Path | Role | GT? |
| --- | --- | --- | --- |
| BraTS2021 T2 (50 subj) | `data/brats/<ID>/gt.nii.gz` | simulated benchmark | yes (isotropic 1mm) |
| Nigerian GRAND_TEST T2 | `data/nigerian/<sub>/{axial,coronal,sagittal}.nii.gz` | real low-field | no |

Nigerian subgroups: **3-orientation** (sub65, sub79, sub80, sub81) — best conditioned;
**2-orientation** (axial+sagittal: sub01/04/07/18/19/34/39). Field strength: sub01–39 = 0.3T,
sub53–81 = 1.5T.

---

## 2. Experiment summary

| ID | Experiment | Baselines/arms | Needs build | Priority |
| --- | --- | --- | --- | --- |
| E1 | BraTS multi-subject benchmark | muon, adam, classical, trilinear | trilinear | ★★★ must |
| E2 | Slice-thickness sweep (2/4/5/6 mm) | all | sim configs | ★★★ |
| E3 | #stacks 2 vs 3 orthogonal | all | sim configs | ★★ |
| E4 | Motion robustness | muon, adam, classical | (have motion sim) | ★★ |
| E5 | Noise/SNR robustness (low-field) | all | sim configs | ★★ |
| E6 | Muon-vs-Adam ablation + rank mechanism | muon, adam | (have) | ★★★ must |
| E7 | Real Nigerian qualitative recon | muon, adam, classical | (have) | ★★★ must |
| E8 | Leave-one-stack-out (real quantitative) | all | LOSO harness | ★★★ must |
| E9 | Cross-field-strength 0.3T vs 1.5T | all | (aggregation) | ★★ |
| E10 | NeSVoR SOTA baseline | nesvor | build NeSVoR | ★★ reviewers expect |
| E11 | Segmentation preservation (clinical) | all | seg-Dice harness | ★★ high value |
| E12 | Efficiency / compute table | all | (have profiling) | ★★ |

---

## 3. Infrastructure status

- [x] **Trilinear baseline** — `methods/trilinear/run.py` (in `brats.yaml`/`nigerian.yaml`).
- [x] **Leave-one-stack-out harness** — `scripts/leave_one_stack_out.py` (predicts held-out
      stack via the PSF operator; scale-invariant PSNR/SSIM). The rigorous no-GT metric.
- [x] **`simulate_batch --set`** for thickness/SNR sweeps + `configs/simulate/twostack.yaml`.
- [x] **Rigid stack registration** — `scripts/register_stacks.py` (SimpleITK Mattes-MI,
      updates affine only, no resampling). Wired into `run_africai.sh` via `REGISTER=true`.
- [x] **No-GT qualitative panels** — `scripts/qualitative_figure.py`.
- [x] **Auto-discovery** — `run_benchmark.py --subjects-root <dir> --n N --results-dir <dir>`.
- [ ] **NeSVoR** — build from source (`methods/nesvor/README.md`) or run on x86+CUDA; adapter
      auto-skips if absent.
- [ ] **Segmentation-Dice** — `scripts/seg_eval.py` (TODO): Dice of a fixed segmenter on
      GT / LR / SRR; BraTS `*_seg.nii.gz` labels available in the source dir.
- [ ] **Aggregation mean±std / per-field-strength** — extend `benchmark/aggregate.py`.

> **⚠️ Fixed bug (was invalidating results):** `load_stacks_dir` used to load `gt.nii.gz`
> as a 4th input stack (GT leakage) — now excludes `gt`/`recon*`. **All earlier simulated
> numbers are void**; `run_africai.sh` re-runs everything cleanly. Regression test added.

---

## 4. Experiment recipes

### E1 — BraTS multi-subject benchmark  ★★★
**Goal:** mean±std PSNR/SSIM/NRMSE/NCC across N subjects; the controlled, GT-backed result.
**Prereq:** simulate stacks for all subjects.
```bash
conda run -n dev python scripts/simulate_batch.py --root data/brats --config configs/simulate/default.yaml
# edit configs/experiment/brats.yaml to list N subjects (or generate from data/brats/subjects.txt)
conda run -n dev python benchmark/run_benchmark.py --config configs/experiment/brats.yaml
conda run -n dev python benchmark/aggregate.py --results results/brats
```
**Outputs:** `results/brats/{results,summary}.csv`, `figures/`. **Cost:** ~15–20 min/INR-run
(free GPU) → scope N to what compute allows (start N=5, scale). Add `trilinear` once built.

### E2 — Slice-thickness sweep  ★★★
**Goal:** show the method's advantage grows with anisotropy (2→6 mm). Table/curve PSNR vs thickness.
```bash
for t in 2 4 5 6; do
  conda run -n dev python scripts/simulate_batch.py --root data/brats \
     --config configs/simulate/default.yaml --overwrite --n 5 \
     # OR a per-thickness config; simplest: make configs/simulate/thick${t}.yaml with thickness_mm: $t
  conda run -n dev python benchmark/run_benchmark.py --config configs/experiment/brats.yaml \
     # point results_dir at results/brats_thick${t}
done
```
**Needs:** per-thickness sim configs + per-thickness results dirs. **Figure:** PSNR-vs-thickness lines.

### E3 — Number of stacks (2 vs 3)  ★★
**Goal:** match the real Nigerian regime (most subjects have 2 orientations). Simulate axial+sagittal
only vs 3 orthogonal.
**Needs:** `configs/simulate/twostack.yaml` (drop coronal). Run E1 pipeline on each.

### E4 — Motion robustness  ★★
**Goal:** PSNR vs inter-stack motion magnitude; low-field scans move.
**Needs:** sim configs varying `motion.max_rot_deg`/`max_trans_mm` (0/3/6/10°). Note: current recon
assumes KNOWN motion — to be honest, either (a) report with known motion (upper bound) or (b) add
`register_stacks.py` and report with estimated motion.

### E5 — Noise / SNR robustness  ★★
**Goal:** low-field = low SNR; show graceful degradation. Sweep `snr ∈ {10,20,30}`.
**Needs:** sim configs varying `snr`.

### E6 — Muon-vs-Adam ablation + rank mechanism  ★★★
**Goal:** the core scientific claim + the mechanism figure (stable/effective rank curves).
Already have n=1: **Muon 31.71 > classical 29.73 > Adam 27.83**, ranks Adam 129→31 vs Muon 129→64.
```bash
# per subject, both arms already run in E1; extract rank history from profile.json
conda run -n dev python scripts/compare_figure.py --subject-dir data/brats/<ID> --results-dir results/brats/<ID> --out results/brats/compare_<ID>.png
# rank-evolution figure is produced by benchmark/aggregate.py (stable_rank_muon_vs_adam.png)
```
**muon_lr note:** at 1500 iters, 3e-2 > 1e-2 > 3e-3 (faster convergence); default kept at 1e-2
(validated to 31.71 @4000 iters). Consider 3e-2 for a lower-iter/faster-compute setting.

### E7 — Real Nigerian qualitative reconstruction  ★★★
**Goal:** the headline real-data figures (thick stacks → isotropic recon), per method, per field strength.
```bash
conda run -n dev python benchmark/run_benchmark.py --config configs/experiment/nigerian.yaml
# qualitative panels (no GT -> compare methods side by side; show input stack + recon orthoviews)
```
**Needs:** a no-GT qualitative figure script (orthoview panels). Uses `per_stack` normalization (set).
**Whole-head real data:** Nigerian scans are NOT skull-stripped, so the INR wastes capacity on
skull/scalp and the normalizer spans the whole FOV. Use `run_nigerian_masked.sh` — it SynthStrips
(via Singularity) a quick trilinear reference into a brain mask, then reconstructs the **brain ROI**
only (`roi_mask` crops the grid + drops non-brain samples), which sharply improves visual quality
and matches the skull-stripped BraTS setup.

### E8 — Leave-one-stack-out (real quantitative)  ★★★
**Goal:** rigorous quantitative metric on real data without GT. Reconstruct from K−1 stacks, predict
the held-out stack, report PSNR/SSIM on it.
```bash
conda run -n dev python scripts/leave_one_stack_out.py --stacks data/nigerian/sub65 \
   --method inr_muon --config configs/default.yaml --set normalize_stacks=per_stack
```
**Needs:** `scripts/leave_one_stack_out.py` (build). Run for muon/adam/classical on 3-orientation subjects.

### E9 — Cross-field-strength (0.3T vs 1.5T)  ★★
**Goal:** multi-site/domain-shift story AfriCAI values. Group E7/E8 results by field strength.
**Needs:** aggregation grouping (field strength in stack sidecar `meta`).

### E10 — NeSVoR SOTA baseline  ★★
**Goal:** compare against the INR-SVR SOTA reviewers expect.
```bash
# build NeSVoR (methods/nesvor/README.md) then:
conda run -n dev python methods/nesvor/run.py --stacks data/brats/<ID> --gt data/brats/<ID>/gt.nii.gz --out results/brats/<ID>/nesvor
```
**On the server:** `sbatch run_nesvor.sh` — pulls `docker://junshenxu/nesvor` via
Singularity/Apptainer (A30 is x86_64/Ampere, so the prebuilt image works), runs the
adapter into `results/africai/brats_5mm`, and re-aggregates so NeSVoR joins Table 1.

### E11 — Segmentation preservation (clinical relevance)  ★★
**Goal:** show SRR recovers downstream accuracy lost to thick slices. Dice of a fixed segmenter on
GT vs LR-upsampled vs each SRR (BraTS tumor labels available in source dir).
**Needs:** `scripts/seg_eval.py` + a segmenter (e.g. an existing nnU-Net/HD-BET, or BraTS `_seg.nii.gz`
as reference for tumor overlap).

### E12 — Efficiency / compute table  ★★
**Goal:** deployability argument. Recon time, peak GPU mem, params, throughput, energy per method —
already captured in every `profile.json`.
```bash
conda run -n dev python benchmark/aggregate.py --results results/brats   # summary.csv has the compute columns
```
**INR speedups implemented:** batched multi-stack forward + **bf16 AMP** (config `amp: true`).
Micro-benchmark (512×6 FFN, real batch, GB10): eager 514 ms/iter → **AMP 116 ms/iter (~4.4×)**.
`torch.compile` is *disabled* — slower on GB10 (few SMs → Inductor overhead). AMP costs
**~0.5 dB PSNR** (phantom: fp32 30.39 vs bf16 29.92) and speeds up **only the INR**, so default is
`amp: false` for headline numbers; use `--set amp=true` for exploratory/robustness sweeps.

---

## 5. Paper mapping

- **Fig 1** method schematic (stacks → PSF forward → INR/Muon → isotropic).
- **Fig 2** real Nigerian qualitative (E7) — the money figure.
- **Fig 3** stable-rank mechanism, Muon vs Adam (E6).
- **Fig 4** PSNR-vs-thickness / robustness curves (E2, E4, E5).
- **Table 1** BraTS benchmark, mean±std, all methods (E1).
- **Table 2** real-data leave-one-stack-out + cross-field-strength (E8, E9).
- **Table 3** efficiency/compute (E12).
- **Table 4** segmentation Dice (E11) if time.

## 6. Suggested run order (given shared GPU / ~1 week)
1. Build trilinear + LOSO harness (CPU/quick).
2. E1 on N=5–10 BraTS (overnight when GPU free) + E6 (free, from E1 outputs).
3. E7 + E8 on the 4 three-orientation Nigerian subjects (the AfriCAI headline).
4. E2 thickness sweep on N=5.
5. E9 aggregation; E12 table (free).
6. If time: E10 NeSVoR (x86 box), E11 segmentation, E4/E5 robustness.
