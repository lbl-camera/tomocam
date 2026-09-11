#!/usr/bin/env bash
#SBATCH --job-name=tomocam-profile
#SBATCH -A <ACCOUNT>
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks=1
#SBATCH -t 00:30:00
#SBATCH -o profile_perlmutter_%j.out
#SBATCH -e profile_perlmutter_%j.err
#
# NERSC/Perlmutter wrapper around profile_tb86.sh: runs the same
# Nsight-Systems timeline analysis (split-Bregman, 1 outer iteration) for
# tb86.json and drisdell.json, back to back, as a single GPU-node job.
#
# Edit the "-A <ACCOUNT>" line above (or pass -A on the sbatch command line)
# before submitting. Assumes tomocam is already built at $BUILD_DIR (default
# ./build) -- this script does not configure/build it.
#
# Submit:   sbatch profile_perlmutter.sh
# Test interactively first, e.g.:
#   salloc -A <account> -C gpu -q interactive -N 1 --gpus-per-node=4 -t 00:30:00
#   ./profile_perlmutter.sh          # runs directly, no sbatch needed
#
# Env overrides: REPO_ROOT, BUILD_DIR, NUM_ITERS (default 1), CONFIGS
# (default: tb86.json drisdell.json)

set -uo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
BUILD_DIR="${BUILD_DIR:-$REPO_ROOT/build}"
NUM_ITERS="${NUM_ITERS:-1}"
CONFIGS=(${CONFIGS:-tb86.json drisdell.json})

cd "$REPO_ROOT" || exit 1

# --- environment -------------------------------------------------------
# Perlmutter GPU nodes need the cuda module for the driver/runtime libs and
# nsys; adjust/remove if your login shell already has these via .bashrc.
module load cudatoolkit 2>/dev/null || true
module load nsight-systems 2>/dev/null || true

if ! command -v nsys >/dev/null 2>&1; then
    echo "error: nsys not found on PATH even after 'module load cudatoolkit/nsight-systems'" >&2
    echo "       run 'module avail nsight-systems' / 'module avail cudatoolkit' and adjust this script" >&2
    exit 1
fi

if [[ ! -x "$BUILD_DIR/mbir_bregman_recon" ]]; then
    echo "error: $BUILD_DIR/mbir_bregman_recon not found -- build tomocam on this node first" >&2
    exit 1
fi

TMP_CFG_DIR=$(mktemp -d)
trap 'rm -rf "$TMP_CFG_DIR"' EXIT

for cfg in "${CONFIGS[@]}"; do
    src="$REPO_ROOT/$cfg"
    if [[ ! -f "$src" ]]; then
        echo "warning: $src not found, skipping" >&2
        continue
    fi

    label="$(basename "$cfg" .json)_${NUM_ITERS}iter"
    patched="$TMP_CFG_DIR/${label}.json"
    # override MBIR.num_iters -> $NUM_ITERS, leave everything else (in
    # particular "filename", which you've already pointed at the NERSC
    # copy of the data) untouched
    sed -E "s/(\"num_iters\"[[:space:]]*:[[:space:]]*)[0-9]+/\1${NUM_ITERS}/" \
        "$src" > "$patched"

    echo "=========================================================="
    echo "== $cfg  ->  num_iters=$NUM_ITERS  (label: $label)"
    echo "=========================================================="

    BUILD_DIR="$BUILD_DIR" \
    CONFIG="$patched" \
    LABEL="$label" \
        "$REPO_ROOT/profile_tb86.sh"
done

echo "all runs complete; see $REPO_ROOT/profiling/*_${NUM_ITERS}iter_*/"
