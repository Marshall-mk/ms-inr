#!/bin/bash -l
#SBATCH --output=/users/%u/nesvor_%j.out
#SBATCH --error=/users/%u/nesvor_%j.err
#SBATCH --job-name=nesvor
#SBATCH --partition=biomed_a30_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
# ---------------------------------------------------------------------------
# Run the NeSVoR (SOTA INR SVR) baseline via the official Docker image, executed
# with Singularity/Apptainer (HPC-safe; no root/Docker daemon needed). Results are
# written INTO the existing results/africai/brats_5mm tree so NeSVoR slots into the
# same tables; the dir is re-aggregated at the end. Logs -> /users/<user>/nesvor_<jobid>.out
#
#   sbatch run_nesvor.sh                    # submit from the repo root
#
# If compute nodes have no internet, pre-pull the image on a login node first:
#   module load apptainer 2>/dev/null || module load singularity
#   apptainer pull nesvor.sif docker://junshenxu/nesvor:latest
# ---------------------------------------------------------------------------
# HPC module loads are optional (guarded) so this also runs on a personal machine.
module load cuda 2>/dev/null || true
module load anaconda3/2022.10-gcc-13.2.0 2>/dev/null || true
module load apptainer 2>/dev/null || module load singularity 2>/dev/null || true
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate "${CONDA_ENV:-dev}"

set -u; shopt -s nullglob; export PYTHONUNBUFFERED=1
cd "${MSINR_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}}" || { echo "FATAL cd"; exit 1; }
echo "Project root: $(pwd)"

N_BRATS=${N_BRATS:-15}
RES=${RES:-results/africai/brats_5mm}
IMG=${NESVOR_IMAGE:-docker://junshenxu/nesvor:latest}
SIF=${NESVOR_SIF:-$PWD/nesvor.sif}
DO_BRATS=${DO_BRATS:-true}
DO_NIGERIAN=${DO_NIGERIAN:-true}
NIG_ROOT=${NIG_ROOT:-data/nigerian_reg}
NIG_3ORI=${NIG_3ORI:-"sub65 sub79 sub80 sub81"}
mkdir -p results/africai/logs

# --- container runtime ----------------------------------------------------
CRT=$(command -v apptainer || command -v singularity || true)
[ -z "$CRT" ] && { echo "FATAL: no apptainer/singularity on PATH (module load apptainer?)"; exit 1; }
echo "container runtime: $CRT"
# keep the (multi-GB) image cache + build tmp off quota-limited home / tiny /tmp
export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-$PWD/.apptainer_cache}"
export SINGULARITY_CACHEDIR="$APPTAINER_CACHEDIR"
export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-$PWD/.apptainer_tmp}"
export SINGULARITY_TMPDIR="$APPTAINER_TMPDIR"
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"

# --- image (pull once) ----------------------------------------------------
if [ ! -f "$SIF" ]; then
  echo "Pulling $IMG -> $SIF ... (if this HANGS, the compute node has no internet ->"
  echo "  cancel, and on a LOGIN NODE run: bash scripts/pull_images.sh )"
  timeout 1800 "$CRT" pull "$SIF" "$IMG" || {
    echo "FATAL: image pull failed/timed out. Pre-pull on a login node:"
    echo "  bash scripts/pull_images.sh"; exit 1; }
fi
"$CRT" exec --nv "$SIF" nesvor --help >/dev/null 2>&1 \
  && echo "nesvor CLI reachable + GPU visible in container" \
  || echo "WARN: 'nesvor --help' failed in container (continuing; per-subject runs may still work)"

# --- wrapper so methods/nesvor/run.py finds `nesvor` on PATH --------------
# On WSL the NVIDIA libs live in /usr/lib/wsl/lib, which --nv doesn't auto-bind.
WSL_ARGS=""
if [ -d /usr/lib/wsl/lib ]; then
  WSL_ARGS="--bind /usr/lib/wsl:/usr/lib/wsl"
  export SINGULARITYENV_LD_LIBRARY_PATH="/usr/lib/wsl/lib:${SINGULARITYENV_LD_LIBRARY_PATH:-}"
  export APPTAINERENV_LD_LIBRARY_PATH="$SINGULARITYENV_LD_LIBRARY_PATH"
fi
mkdir -p .nesvor_bin
cat > .nesvor_bin/nesvor <<EOF
#!/bin/bash
exec $CRT exec --nv $WSL_ARGS --bind "$PWD:$PWD" --pwd "$PWD" "$SIF" nesvor "\$@"
EOF
chmod +x .nesvor_bin/nesvor
export PATH="$PWD/.nesvor_bin:$PATH"
echo "nesvor wrapper -> $(command -v nesvor)"

# fail fast if the GPU isn't reachable inside the container
if ! "$CRT" exec --nv $WSL_ARGS "$SIF" python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  echo "WARN: CUDA not available inside the container. If host 'nvidia-smi' works this is the"
  echo "      WSL lib-bind issue; otherwise this machine has no usable GPU -> run on the HPC"
  echo "      (scp nesvor.sif to the cluster and sbatch run_nesvor.sh there)."
fi

# ===== BraTS simulated (with GT) — NeSVoR into the brats_5mm tables ========
if [ "$DO_BRATS" = true ]; then
  subs=$(ls -d data/brats/*/ 2>/dev/null | head -n "$N_BRATS")
  for d in $subs; do
    s=$(basename "$d")
    [ -n "$(ls "$d"/stack_*.nii.gz 2>/dev/null)" ] || { echo "skip $s (no stacks)"; continue; }
    out="$RES/$s/nesvor"
    if [ -f "$out/recon.nii.gz" ]; then echo "== $s: nesvor already done =="; continue; fi
    echo "== NeSVoR (BraTS): $s =="
    python methods/nesvor/run.py --stacks "data/brats/$s" --gt "data/brats/$s/gt.nii.gz" \
      --out "$out" --set output_resolution=1.0 \
      2>&1 | tee -a results/africai/logs/nesvor_brats5mm.log || echo "  ($s failed; continuing)"
  done
  python benchmark/aggregate.py --results "$RES"
  echo "NeSVoR BraTS complete -> $RES (results.csv / summary.csv updated)."
fi

# ===== Real Nigerian (no GT) — reconstruction + LOSO =======================
if [ "$DO_NIGERIAN" = true ]; then
  echo "Nigerian root: $NIG_ROOT ($(ls -d "$NIG_ROOT"/sub*/ 2>/dev/null | wc -l) subjects)"
  # E7: one NeSVoR recon per subject, then refresh the qualitative panel (now incl. nesvor)
  for d in "$NIG_ROOT"/sub*/; do
    s=$(basename "$d")
    out="results/africai/nigerian/$s/nesvor"
    if [ ! -f "$out/recon.nii.gz" ]; then
      echo "== NeSVoR (Nigerian): $s =="
      python methods/nesvor/run.py --stacks "$NIG_ROOT/$s" --out "$out" \
        --set output_resolution=1.0 iso_mm=1.0 \
        2>&1 | tee -a results/africai/logs/nesvor_nigerian.log || echo "  ($s failed; continuing)"
    fi
    python scripts/qualitative_figure.py --results-dir "results/africai/nigerian/$s" \
      --stacks "$NIG_ROOT/$s" \
      --methods trilinear classical_srr inr_adam inr_muon nesvor \
      --out "results/africai/nigerian/qual_${s}.png" || true
  done
  # E8: NeSVoR as a leave-one-stack-out method on the 3-orientation subjects
  for s in $NIG_3ORI; do
    lo="results/africai/loso/${s}_nesvor"
    [ -f "$lo/loso.json" ] && { echo "== LOSO nesvor $s already done =="; continue; }
    echo "== LOSO NeSVoR: $s =="
    python scripts/leave_one_stack_out.py --stacks "$NIG_ROOT/$s" --method nesvor \
      --config configs/default.yaml --set normalize_stacks=per_stack iso_mm=1.0 \
      --out "$lo" 2>&1 | tee -a results/africai/logs/nesvor_loso.log || echo "  ($s failed; continuing)"
  done
fi

echo "run_nesvor.sh complete."
