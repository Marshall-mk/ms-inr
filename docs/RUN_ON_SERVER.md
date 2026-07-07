# Running the full pipeline on the server

Run everything from the **repo root**. Three SLURM jobs produce all results under
`results/africai/`. Real-data reconstruction uses `configs/real_lowfield.yaml`
(regularized: sigma=3, per-stack bias-field, weight decay, looser ROI crop, clamp);
simulated BraTS uses `configs/default.yaml`.

## 0. One-time prerequisites
```bash
git pull
# if the repo was checked out on Windows/WSL, fix line endings:
sed -i 's/\r$//' *.sh scripts/*.sh   2>/dev/null || true

# env: `dev` must have torch(+CUDA), muon, nibabel, scipy, scikit-image, SimpleITK,
# pyyaml, matplotlib. (run_*.sh auto-install muon + SimpleITK if missing.)

# containers: pre-pull on a LOGIN node (compute nodes usually have no internet):
bash scripts/pull_images.sh          # -> nesvor.sif, synthstrip.sif

# data: need data/brats/<ID>/gt.nii.gz and data/nigerian/<sub>/*.nii.gz. If missing:
#   python scripts/prep_brats.py    --src <BraTS_dir>   --n 50 --out data/brats
#   python scripts/prep_nigerian.py --src <Nigerian_dir>       --out data/nigerian

# register the real Nigerian stacks once (CPU, quick) so all 3 jobs are independent:
for d in data/nigerian/sub*/; do s=$(basename "$d"); \
  python scripts/register_stacks.py --stacks "data/nigerian/$s" --out "data/nigerian_reg/$s"; done
```

## 1. Simulated BraTS benchmark + Nigerian LOSO  (the bulk; several hours)
```bash
sbatch run_africai.sh
```
Produces: `results/africai/brats_5mm` (E1, all methods) · `brats_thick{2,4,6}` (E2 thickness
sweep) · `brats_twostack` (E3) · `loso/` (E8, real quantitative) · per-set `results.csv`,
`summary.csv`, `figures/`. Knobs: `N_BRATS` (default 15), `N_THICK` (5), `THICKNESSES`
("2 4 6"), `REGISTER` (true). Resumable (re-`sbatch` continues).

## 2. Brain-ROI masked real reconstruction  (primary real qualitative)
```bash
sbatch run_nigerian_masked.sh
```
SynthStrips each Nigerian subject → brain mask → reconstructs the brain ROI with all methods
using the regularized real config → `results/africai/nigerian_masked/<sub>/<method>` +
`qual_<sub>.png`.

## 3. NeSVoR baseline  (SOTA; with the registration fix)
```bash
sbatch run_nesvor.sh                 # DO_BRATS + DO_NIGERIAN (either can be false)
```
Runs NeSVoR (via Singularity, `--registration none` so it trusts our affines): BraTS → merged
into `brats_5mm` tables; Nigerian recon + LOSO. Knobs: `N_BRATS`, `NIG_ROOT`, `NESVOR_SIF`.

## Bring back
`rsync`/`scp` the whole `results/africai/` tree home (CSVs, figures, `loso/*.json`, recon
NIfTIs, `logs/`). Then we build the paper tables/figures and start writing.

## Notes
- Jobs 1–3 are independent once Nigerian is registered (step 0), so submit them together.
- Logs go to `/users/<user>/<jobname>_<jobid>.out|.err`.
- On a non-SLURM machine, replace `sbatch X` with `bash X` (module loads are guarded).
