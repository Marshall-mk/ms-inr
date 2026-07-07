#!/bin/bash
# Pre-pull the container images ON A LOGIN NODE (compute nodes usually have no
# internet). Run this once from the repo root; the SLURM scripts then find the
# .sif files and skip pulling. NOT an sbatch job -- run it directly:
#     bash scripts/pull_images.sh
set -u
cd "${MSINR_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}" || exit 1
module load apptainer 2>/dev/null || module load singularity 2>/dev/null || true
CRT=$(command -v apptainer || command -v singularity || true)
[ -z "$CRT" ] && { echo "FATAL: no apptainer/singularity (module load apptainer?)"; exit 1; }
echo "runtime: $CRT   in: $(pwd)"

# keep the multi-GB cache + build tmp off a quota-limited home dir
export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-$PWD/.apptainer_cache}"
export SINGULARITY_CACHEDIR="$APPTAINER_CACHEDIR"
export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-$PWD/.apptainer_tmp}"
export SINGULARITY_TMPDIR="$APPTAINER_TMPDIR"
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"

pull() {  # $1=sif  $2=docker uri
  if [ -f "$1" ]; then echo "already have $1"; return 0; fi
  echo "pulling $2 -> $1"
  "$CRT" pull "$1" "$2" && echo "  OK: $1" \
    || echo "  FAILED: $2  (Docker Hub rate limit? retry later, or 'singularity remote login')"
}

pull "$PWD/nesvor.sif"     docker://junshenxu/nesvor:latest
pull "$PWD/synthstrip.sif" docker://freesurfer/synthstrip:latest
echo "done. Now: sbatch run_nesvor.sh   and   sbatch run_nigerian_masked.sh"
