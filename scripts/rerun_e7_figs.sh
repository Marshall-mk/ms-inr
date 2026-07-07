#!/bin/bash -l
#SBATCH --output=/users/%u/rerun_e7_%j.out
#SBATCH --error=/users/%u/rerun_e7_%j.err
#SBATCH --job-name=rerun_e7
#SBATCH --partition=biomed_a30_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=0-02:00:00
# ---------------------------------------------------------------------------
# Rerun ONLY the E7 leftovers after the qualitative_figure + classical-OOM fixes:
#   - rerun classical_srr for any Nigerian subject whose recon is missing (OOM'd)
#   - (re)generate every Nigerian qualitative figure
# CPU work; the GPU partition is just to match your known-good queue. Logs -> your
# home: /users/<user>/rerun_e7_<jobid>.out|.err
#   sbatch scripts/rerun_e7_figs.sh        # submit from the repo root
# ---------------------------------------------------------------------------
module load cuda
module load anaconda3/2022.10-gcc-13.2.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate dev

set -u
shopt -s nullglob
export PYTHONUNBUFFERED=1
cd "${MSINR_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}}" \
  || { echo "FATAL: cannot cd to project root"; exit 1; }
echo "Project root: $(pwd)"

NIG_ROOT=${NIG_ROOT:-data/nigerian_reg}
NIG_SET="normalize_stacks=per_stack iso_mm=1.0"
n=$(ls -d "$NIG_ROOT"/sub*/ 2>/dev/null | wc -l)
echo "Nigerian root: $NIG_ROOT ($n subjects)"
if [ "$n" -eq 0 ]; then
  echo "FATAL: no subjects under $NIG_ROOT — run register (REGISTER=true) in run_africai.sh first,"
  echo "       or set NIG_ROOT=data/nigerian to use the unregistered stacks."
  exit 1
fi

for d in "$NIG_ROOT"/sub*/; do
  s=$(basename "$d")
  out="results/africai/nigerian/$s"
  mkdir -p "$out"
  if [ ! -f "$out/classical_srr/recon.nii.gz" ]; then
    echo "== rerun classical_srr: $s =="
    python methods/classical_srr/run.py --stacks "$NIG_ROOT/$s" \
      --out "$out/classical_srr" --set $NIG_SET reg_lambda=0.05 cg_maxiter=200 \
      || echo "  (classical still failed for $s; if OOM, retry with iso_mm=1.5)"
  else
    echo "== classical_srr already present: $s =="
  fi
  echo "== qualitative figure: $s =="
  python scripts/qualitative_figure.py --results-dir "$out" \
    --stacks "$NIG_ROOT/$s" --out "results/africai/nigerian/qual_${s}.png" || true
done
echo "DONE -> results/africai/nigerian/qual_*.png"
