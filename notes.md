# Paper Notes — Rank-Optimized INRs for Multi-Stack SRR of (Low-Field) Adult MRI

Working notes for the AfriCAI 2026 paper. Confidence tags: **[SOLID]** (from the clean
server benchmark), **[PRELIM]** (from small dev A/Bs, needs a clean masked re-run),
**[METHOD]** (implementation detail), **[TODO]** (still to run/confirm).

---

## 1. Thesis & positioning
- **Method:** per-scan, **training-free / self-supervised** super-resolution reconstruction
  (SRR) of multi-stack MRI with a coordinate INR, trained with the **Muon optimizer**
  (high-rank/near-orthogonal updates; McGinnis et al., arXiv:2512.14366). Novel piece =
  coupling rank-optimized INR training with a 3D multi-stack PSF forward model (the
  NeSVoR/IREM setting), which the paper never tested.
- **AfriCAI angle:** low-field scanners (our Nigerian 0.3T/1.5T data) produce thick-slice,
  anisotropic scans; SRR recovers isotropic volumes **without new hardware**. Training-free =
  no labels, no cohort, no domain shift → directly answers African data scarcity. Multi-site
  (0.3T + 1.5T) built in.
- **One-liner:** *Training-free, scan-specific super-resolution that makes low-field African
  MRI isotropic and high-fidelity, using rank-optimized INRs — validated on real 0.3T/1.5T data.*

## 2. Datasets
- **BraTS2021 T2** (skull-stripped, 1mm iso, 240×240×155): simulated benchmark with GT. Used
  first 15–50 subjects.
- **Nigerian GRAND_TEST T2** (real, whole-head, not skull-stripped): 0.3T (axial+sagittal, ~7mm)
  and 1.5T (3 orthogonal, ~5mm). 3-orthogonal subjects: **sub65, sub79, sub80, sub81**.
- **Simulation protocol** (NeSVoR adult regime): 3 orthogonal stacks, anisotropic Gaussian PSF
  (through-plane FWHM = slice thickness, σ = t/2.3552), rigid inter-stack motion ±3mm/±6°,
  Rician noise, thickness sweep {2,4,5,6} mm. Metrics PSNR/SSIM/NRMSE/NCC, brain-masked.

## 3. Headline results (simulated BraTS, GT)  **[SOLID]**
**5mm, N=15, matched 512×6 INR (PSNR / SSIM):**

| Method | PSNR | SSIM | NCC |
|---|---|---|---|
| **inr_muon** | **24.78** | **0.947** | **0.920** |
| inr_adam | 23.07 | 0.921 | 0.881 |
| classical_srr (CG LS) | 18.21 | 0.881 | 0.893 |
| trilinear | 17.82 | 0.754 | 0.742 |

→ **Muon-INR is best: +1.7 dB over Adam, +6.6 dB over classical.** Muon > Adam **everywhere**.

**Slice-thickness robustness (N=5), PSNR muon / adam / classical / trilinear:**

| thickness | muon | adam | classical | tri |
|---|---|---|---|---|
| 2 mm | 26.6 | 24.6 | **25.5** | 22.4 |
| 4 mm | **25.5** | 23.7 | 19.7 | 19.4 |
| 5 mm | **24.8** | 23.1 | 18.2 | 17.8 |
| 6 mm | **24.1** | 22.5 | 16.5 | 17.7 |

→ **Key figure.** Classical **collapses** with thickness (25.5→16.5); Muon-INR **degrades
gracefully** (26.6→24.1). At thin slices (2mm) classical is competitive; the thicker (i.e.
more realistic low-field) the slices, the bigger the INR advantage.

**Two-stack (axial+sagittal, the common Nigerian 0.3T case), N=5:** Muon 25.24 > Adam 23.41 >
classical 19.40 > tri 18.36. → INR advantage holds (and grows) with fewer stacks.

## 4. Mechanism — Muon vs Adam  **[TODO re-confirm on clean run]**
- Story: Adam lets hidden-layer **stable rank collapse** during training; Muon **preserves**
  high stable rank → better high-frequency fidelity (the paper's claim, in the SRR setting).
- Early observation (Adam 129→31 vs Muon 129→64) is indicative but from a preliminary run;
  **re-extract stable/effective-rank curves from the final benchmark** for the mechanism figure.

## 5. Real low-field data (Nigerian)
- **LOSO (leave-one-stack-out; predict held-out stack, no GT)** **[SOLID]**, mean PSNR:
  classical ≳ muon > adam > trilinear; e.g. sub80: cls 25.06 / muon 24.28 / adam 24.10 / tri 23.96.
  - **Muon > Adam holds on real data too.**
  - **Caveat to state plainly:** LOSO structurally favors classical LS — least-squares *directly*
    minimizes stack-reprojection error, which is exactly what LOSO measures. It's a
    *consistency* proxy, not isotropic fidelity. Report it, but don't over-read classical's edge.
- **Qualitative (whole-head dev A/B, sub65 & sub01)** **[PRELIM]**:
  - **3-orthogonal (sub65) reconstructs well even at baseline** — clear ventricles/cortex/
    cerebellum/brainstem. **2-orthogonal (sub01) is the hard, under-determined case.** → lead
    real-data results with sub65/79/80/81.
  - classical shows **checkerboard/grid artifacts** on 2-stack sub01 (null-space of the
    under-determined operator) → needs higher `reg_lambda`.

## 6. Central real-data finding (the interesting one)  **[PRELIM → confirm]**
**Muon's high-frequency / high-rank bias is an asset on clean simulated data but a liability
on noisy real low-field data** — it fits the noise. Concretely, on real data the INR outputs
were **grainy (high-freq noise), not blurry** (blur would indicate under-capacity), and
**Muon was noisier than Adam**.
- **Effective resolution must be dialed DOWN for real data.** Things that raise effective
  frequency and amplify noise on real data: high `sigma`, fine `iso_mm`, many `iters`, and —
  importantly — **tight ROI cropping** (cropping the grid to the brain bbox shrinks the
  normalizer domain, so the same Fourier features resolve finer detail → more noise). The very
  masking meant to help *amplified* the graininess on sub01.
- **Fixes that help (subtle but consistent):** per-stack **bias-field model**, lower `sigma`
  (6→3), **weight decay** (1e-4), output clamp (≥0). Recommended real-data config:
  `sigma=3, bias_field=poly, weight_decay=1e-4, muon_weight_decay=1e-4, clamp_min=0`, looser
  ROI margin (~15–20mm), `iso≈1.0`, moderate iters.
- **Why this is a good paper point:** it's a genuine, reportable nuance about *applying*
  rank-optimized INRs to real African low-field MRI — not just "our net is too small."

## 7. Capacity / architecture  **[SOLID]**
- On BraTS, **depth > width**: 6 layers @ 512 (~1.3M params) → 22.9dB direct-fit vs ~19dB for
  4 layers; wider-but-shallow (1024×4) was *worse* than deeper. Use **512×6**.
- Under-capacity presents as **blur**; the real-data problem was **noise**, so it is *not*
  primarily capacity (adding layers/frequency made noisy real data worse, not better).

## 8. Baselines & NeSVoR  **[METHOD/TODO]**
- Baselines: trilinear (lower bound), classical LS-SRR (CG, same PSF operator), Adam-INR
  (ablation), Muon-INR (ours), **NeSVoR** (SOTA INR SVR).
- **NeSVoR gotcha:** its fetal-trained **SVoRT registration mis-registers adult stacks**
  (similarity ~0.41) and reconstructs in a re-registered pose → misaligned with GT → garbage
  metric (PSNR 4.89). **Fix: `--registration none`** so it trusts our (correct/pre-registered)
  affines. [TODO] re-run; if still low on adult, that's a legitimate finding (fetal-specialized).
- Runs on the HPC via Singularity (official x86 image on A30); needs `--registration none`.

## 9. Efficiency / compute  **[SOLID]**
- Per subject: INR ~10 min (A30) / ~15–20 min (GB10 free); classical ~8s; trilinear ~4s;
  NeSVoR ~15 min. INR peak GPU mem ~2.8–3.7 GB, 1.3M params.
- **bf16 AMP ≈ 2.7–4.4× faster** but costs ~0.5 dB and slows *only* the INR → keep OFF for
  headline numbers, ON for exploratory sweeps. `torch.compile` is *slower* on GB10 (few SMs).

## 10. Method / implementation details  **[METHOD]**
- INR: `relu_ffn`, hidden 512 × 6 layers, Fourier `mapping_size=256`, `sigma` 6 (clean) / 3 (real).
- Muon on hidden 2D weight matrices only; first/last + biases + bias-field params → aux Adam.
- Forward model: anisotropic Gaussian PSF slice sampler, per-stack world offsets; loss = MSE on
  observed slices. Intensity normalization: `global` (simulated) / `per_stack` (real multi-scanner).
- Bias-field: per-stack multiplicative `exp(poly(x_norm))` (degree-2), Adam-optimized; absorbs
  inter-stack intensity disagreement so the field f stays clean.
- Real data: rigid stack registration (SimpleITK Mattes-MI, affine-only) + SynthStrip brain mask
  (Singularity) → ROI crop + sample mask + output mask.

## 11. Methodology (for the methods section)  **[METHOD]**
- Metrics are brain-masked; ROI-cropped recons are resampled onto the GT grid before scoring.
- Simulated stacks are strictly held separate from the GT volume (GT is never an input).

## 12. Open items before writing  **[TODO]**
- [ ] Clean **full re-run on the server** with the current pipeline (ROI mask+output-mask,
      bias-field, NeSVoR `--registration none`) and the improved real-data config.
- [ ] Stable/effective-rank **mechanism figure** from the clean run.
- [ ] **Masked** Nigerian recon (sub65/79/80/81) with the regularized real config; confirm the
      noise-amplification hypothesis and that bias-field helps under masking.
- [ ] NeSVoR numbers after the registration fix (BraTS + Nigerian LOSO).
- [ ] Multi-subject stats (mean±std), per-field-strength (0.3T vs 1.5T) split.
- [ ] (Optional) hash-grid encoder + Muon; segmentation-Dice downstream task; radiologist read.

## 13. Suggested paper figures/tables
- **F1** method schematic (stacks → PSF forward → INR/Muon → isotropic).
- **F2** thickness-robustness curves (§3) — the money figure.
- **F3** Muon-vs-Adam stable-rank mechanism (§4).
- **F4** real Nigerian qualitative (masked, 3-orthogonal) + the noise/regularization point (§6).
- **T1** BraTS benchmark mean±std, all methods incl. NeSVoR (§3).
- **T2** real-data LOSO + cross-field-strength (§5).
- **T3** efficiency/compute (§9).
