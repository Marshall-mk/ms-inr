#!/bin/bash -l
#SBATCH --output=/users/%u/%j.out
#SBATCH --error=/users/%u/%j.err
#SBATCH --job-name=africai
#SBATCH --partition=biomed_a30_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
# ---------------------------------------------------------------------------
# AfriCAI 2026 experiment suite for ms-inr. Runs the full benchmark and saves all
# results under results/ (bring these back for paper writing). Adjust the knobs
# below and the SBATCH partition/time to your cluster. See docs/AFRICAI_EXPERIMENTS.md.
# ---------------------------------------------------------------------------
module load cuda
module load anaconda3/2022.10-gcc-13.2.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate dev

# ---- ensure required deps (install only if missing) ----------------------
python -c "import muon" 2>/dev/null || pip install --quiet "git+https://github.com/KellerJordan/Muon"
python -c "import SimpleITK" 2>/dev/null || pip install --quiet SimpleITK   # stack registration
python -c "import muon, SimpleITK; print('deps OK: muon + SimpleITK importable')" || {
  echo "FATAL: muon/SimpleITK install failed"; exit 1; }

set -u
shopt -s nullglob        # empty sub*/ globs expand to nothing, not a literal string
export PYTHONUNBUFFERED=1

# Under SLURM, $0 is a COPY of this script in the spool dir (/var/lib/slurm/...),
# so `dirname $0` is NOT the repo. Use the submit dir. Override with MSINR_ROOT if
# you sbatch from elsewhere.
PROJECT_ROOT="${MSINR_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}}"
cd "$PROJECT_ROOT" || { echo "FATAL: cannot cd to project root '$PROJECT_ROOT'"; exit 1; }
echo "Project root: $(pwd)"

# fail fast with a clear message if the repo or data isn't where we expect
if [ ! -f benchmark/run_benchmark.py ]; then
  echo "FATAL: benchmark/run_benchmark.py not found in $(pwd)."
  echo "  sbatch this script FROM the ms-inr repo root, or set MSINR_ROOT=/path/to/ms-inr."
  exit 1
fi
if [ ! -d data/brats ] && [ -z "${BRATS_SRC:-}" ]; then
  echo "WARNING: data/brats missing and BRATS_SRC unset -> BraTS experiments will be empty."
fi

# ---- knobs ---------------------------------------------------------------
N_BRATS=${N_BRATS:-15}                 # BraTS subjects for the main benchmark (E1)
N_THICK=${N_THICK:-5}                   # subjects for the thickness/2-stack sweeps
THICKNESSES=${THICKNESSES:-"2 4 6"}     # E2 slice thicknesses (5mm is the E1 default)
BRATS_SRC=${BRATS_SRC:-""}              # optional: source BraTS dir (prep if data/brats missing)
NIGERIAN_SRC=${NIGERIAN_SRC:-""}        # optional: source Nigerian dir
NIG_3ORI=${NIG_3ORI:-"sub65 sub79 sub80 sub81"}   # 3-orientation subjects for LOSO
AMP_SWEEP=${AMP_SWEEP:-true}            # use bf16 AMP for the big sweeps (E2/E3/Nigerian)
REGISTER=${REGISTER:-true}              # rigidly co-register real Nigerian stacks first
SIMCFG=configs/simulate/default.yaml
NIG_ROOT=data/nigerian                  # overwritten below if REGISTER=true
mkdir -p results/africai/logs
run(){ echo -e "\n########## $* ##########"; }

# ---- 0. optional data prep ----------------------------------------------
if [ -n "$BRATS_SRC" ] && [ ! -d data/brats ]; then
  run "PREP BraTS from $BRATS_SRC"
  python scripts/prep_brats.py --src "$BRATS_SRC" --n 50 --contrast t2 --out data/brats
fi
if [ -n "$NIGERIAN_SRC" ] && [ ! -d data/nigerian ]; then
  run "PREP Nigerian from $NIGERIAN_SRC"
  python scripts/prep_nigerian.py --src "$NIGERIAN_SRC" --contrast T2 --out data/nigerian
fi

# ===========================================================================
# E1 — BraTS main benchmark (5mm, all methods, fp32 for headline numbers) + E6
# ===========================================================================
run "E1 simulate BraTS stacks @5mm"
python scripts/simulate_batch.py --root data/brats --config $SIMCFG --n "$N_BRATS"

run "E1 BraTS benchmark (N=$N_BRATS, all methods)"
python benchmark/run_benchmark.py --config configs/experiment/brats.yaml \
  --subjects-root data/brats --n "$N_BRATS" --results-dir results/africai/brats_5mm \
  2>&1 | tee results/africai/logs/e1_brats5mm.log

run "E6 comparison figures (first 3 subjects)"
for d in $(ls -d data/brats/*/ | head -3); do
  s=$(basename "$d")
  python scripts/compare_figure.py --subject-dir "data/brats/$s" \
    --results-dir "results/africai/brats_5mm/$s" \
    --out "results/africai/brats_5mm/compare_${s}.png" || true
done

# ===========================================================================
# E2 — slice-thickness sweep (AMP for speed)
# ===========================================================================
for t in $THICKNESSES; do
  run "E2 thickness=${t}mm (simulate + benchmark, N=$N_THICK)"
  python scripts/simulate_batch.py --root data/brats --config $SIMCFG \
    --n "$N_THICK" --overwrite --set thickness_mm=$t
  python benchmark/run_benchmark.py --config configs/experiment/brats.yaml \
    --subjects-root data/brats --n "$N_THICK" \
    --results-dir "results/africai/brats_thick${t}" --only inr_muon inr_adam classical_srr trilinear \
    2>&1 | tee "results/africai/logs/e2_thick${t}.log"
done
# restore the 5mm stacks for any later step that expects them
python scripts/simulate_batch.py --root data/brats --config $SIMCFG --n "$N_BRATS" --overwrite

# ===========================================================================
# E3 — two-stack (axial+sagittal) regime, matches Nigerian 0.3T
# ===========================================================================
run "E3 two-stack simulate + benchmark (N=$N_THICK)"
python scripts/simulate_batch.py --root data/brats --config configs/simulate/twostack.yaml \
  --n "$N_THICK" --overwrite
python benchmark/run_benchmark.py --config configs/experiment/brats.yaml \
  --subjects-root data/brats --n "$N_THICK" \
  --results-dir results/africai/brats_twostack --only inr_muon inr_adam classical_srr trilinear \
  2>&1 | tee results/africai/logs/e3_twostack.log
python scripts/simulate_batch.py --root data/brats --config $SIMCFG --n "$N_BRATS" --overwrite

# ===========================================================================
# E7 — real Nigerian: (optional) register -> reconstruct -> qualitative figures
# ===========================================================================
AMP_SET=""; [ "$AMP_SWEEP" = "true" ] && AMP_SET="amp=true"
NIG_SET="normalize_stacks=per_stack iso_mm=1.0"
if [ "$REGISTER" = "true" ]; then
  run "Register Nigerian stacks -> data/nigerian_reg"
  NIG_ROOT=data/nigerian_reg
  for d in data/nigerian/sub*/; do
    s=$(basename "$d")
    python scripts/register_stacks.py --stacks "data/nigerian/$s" --out "$NIG_ROOT/$s" \
      2>&1 | tee -a results/africai/logs/register.log || true
  done
fi

run "E7 Nigerian reconstruction (real, no GT) from $NIG_ROOT"
for d in "$NIG_ROOT"/sub*/; do
  s=$(basename "$d")
  for m in trilinear classical_srr inr_adam inr_muon; do
    out="results/africai/nigerian/$s/$m"
    if [ "${m#inr_}" != "$m" ]; then
      python methods/$m/run.py --stacks "$NIG_ROOT/$s" --out "$out" \
        --config configs/default.yaml --set $NIG_SET $AMP_SET || true
    elif [ "$m" = "classical_srr" ]; then
      python methods/classical_srr/run.py --stacks "$NIG_ROOT/$s" --out "$out" \
        --set $NIG_SET reg_lambda=0.05 cg_maxiter=200 || true
    else
      python methods/trilinear/run.py --stacks "$NIG_ROOT/$s" --out "$out" --set $NIG_SET || true
    fi
  done
  python scripts/qualitative_figure.py --results-dir "results/africai/nigerian/$s" \
    --stacks "$NIG_ROOT/$s" --out "results/africai/nigerian/qual_${s}.png" || true
done 2>&1 | tee results/africai/logs/e7_nigerian.log

# ===========================================================================
# E8 — leave-one-stack-out on the 3-orientation Nigerian subjects
# ===========================================================================
run "E8 leave-one-stack-out (real quantitative) from $NIG_ROOT"
for s in $NIG_3ORI; do
  for m in trilinear classical inr_adam inr_muon; do
    python scripts/leave_one_stack_out.py --stacks "$NIG_ROOT/$s" --method "$m" \
      --config configs/default.yaml --set $NIG_SET $AMP_SET \
      --out "results/africai/loso/${s}_${m}" \
      2>&1 | tee -a results/africai/logs/e8_loso.log || true
  done
done

# ===========================================================================
# E12 — aggregate everything (quality + compute tables + figures)
# ===========================================================================
run "E12 aggregate all result sets"
for r in results/africai/brats_5mm results/africai/brats_twostack results/africai/nigerian \
         $(for t in $THICKNESSES; do echo results/africai/brats_thick${t}; done); do
  [ -d "$r" ] && python benchmark/aggregate.py --results "$r" || true
done

run "AfriCAI suite complete. Results under results/africai/"
