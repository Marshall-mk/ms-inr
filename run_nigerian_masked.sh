#!/bin/bash -l
#SBATCH --output=/users/%u/nigmask_%j.out
#SBATCH --error=/users/%u/nigmask_%j.err
#SBATCH --job-name=nigmask
#SBATCH --partition=biomed_a30_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
# ---------------------------------------------------------------------------
# Brain-ROI reconstruction of the real Nigerian data. Whole-head scans make the
# INR waste capacity on skull/scalp and the normalizer span the whole FOV, so we:
#   1. quick isotropic trilinear reference per subject,
#   2. SynthStrip (via Singularity) -> brain mask,
#   3. reconstruct all methods on the BRAIN ROI only (roi_mask crops the grid +
#      drops non-brain samples), and refresh the qualitative panels.
# Results -> results/africai/nigerian_masked/. Logs -> /users/<user>/nigmask_<jobid>.out
#   sbatch run_nigerian_masked.sh        # from the repo root
# ---------------------------------------------------------------------------
# HPC module loads are optional (guarded) so this also runs on a personal machine.
module load cuda 2>/dev/null || true
module load anaconda3/2022.10-gcc-13.2.0 2>/dev/null || true
module load apptainer 2>/dev/null || module load singularity 2>/dev/null || true
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate "${CONDA_ENV:-dev}"

set -u; shopt -s nullglob; export PYTHONUNBUFFERED=1
cd "${MSINR_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}}" || { echo "FATAL cd"; exit 1; }
echo "Project root: $(pwd)"

NIG_ROOT=${NIG_ROOT:-data/nigerian_reg}
RES=${RES:-results/africai/nigerian_masked}
AMP_SET=""; [ "${AMP:-true}" = true ] && AMP_SET="amp=true"
SET="normalize_stacks=per_stack iso_mm=1.0"
mkdir -p results/africai/logs

CRT=$(command -v apptainer || command -v singularity || true)
[ -z "$CRT" ] && { echo "FATAL: no apptainer/singularity"; exit 1; }
export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-$PWD/.apptainer_cache}"
export SINGULARITY_CACHEDIR="$APPTAINER_CACHEDIR"
export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-$PWD/.apptainer_tmp}"
export SINGULARITY_TMPDIR="$APPTAINER_TMPDIR"; mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"
SS_SIF=${SS_SIF:-$PWD/synthstrip.sif}
if [ ! -f "$SS_SIF" ]; then
  echo "Pulling SynthStrip -> $SS_SIF (if it HANGS, pre-pull on a login node: bash scripts/pull_images.sh)"
  timeout 1800 "$CRT" pull "$SS_SIF" docker://freesurfer/synthstrip:latest || {
    echo "FATAL: SynthStrip pull failed/timed out -> bash scripts/pull_images.sh on a login node"; exit 1; }
fi

n=$(ls -d "$NIG_ROOT"/sub*/ 2>/dev/null | wc -l)
echo "Nigerian root: $NIG_ROOT ($n subjects)"; [ "$n" -eq 0 ] && exit 1

for d in "$NIG_ROOT"/sub*/; do
  s=$(basename "$d")
  ref="$NIG_ROOT/$s/_roiref.nii.gz"
  mask="$NIG_ROOT/$s/brain_mask.nii.gz"

  # 1. isotropic reference for stripping (unmasked trilinear)
  if [ ! -f "$ref" ]; then
    python methods/trilinear/run.py --stacks "$NIG_ROOT/$s" --out "$NIG_ROOT/$s/_roitmp" \
      --set $SET && cp "$NIG_ROOT/$s/_roitmp/recon.nii.gz" "$ref"
  fi
  # 2. brain mask via SynthStrip
  if [ ! -f "$mask" ]; then
    echo "== SynthStrip: $s =="
    "$CRT" exec "$SS_SIF" mri_synthstrip -i "$ref" -m "$mask" \
      2>&1 | tee -a results/africai/logs/synthstrip.log \
      || { echo "  SynthStrip failed for $s; skipping"; continue; }
  fi
  # 3. brain-ROI reconstruction, all methods
  for m in trilinear classical_srr inr_adam inr_muon; do
    out="$RES/$s/$m"; [ -f "$out/recon.nii.gz" ] && continue
    echo "== $s / $m (ROI) =="
    if [ "${m#inr_}" != "$m" ]; then
      python methods/$m/run.py --stacks "$NIG_ROOT/$s" --out "$out" \
        --config configs/real_lowfield.yaml --set roi_mask="$mask" $AMP_SET || true
    elif [ "$m" = "classical_srr" ]; then
      python methods/classical_srr/run.py --stacks "$NIG_ROOT/$s" --out "$out" \
        --set $SET roi_mask="$mask" reg_lambda=0.05 cg_maxiter=200 || true
    else
      python methods/trilinear/run.py --stacks "$NIG_ROOT/$s" --out "$out" \
        --set $SET roi_mask="$mask" || true
    fi
  done
  # 4. qualitative panel
  python scripts/qualitative_figure.py --results-dir "$RES/$s" --stacks "$NIG_ROOT/$s" \
    --methods trilinear classical_srr inr_adam inr_muon --out "$RES/qual_${s}.png" || true
  rm -rf "$NIG_ROOT/$s/_roitmp"
done
echo "Brain-ROI Nigerian reconstruction complete -> $RES"
