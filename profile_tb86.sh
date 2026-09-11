#!/usr/bin/env bash
#
# Profile an mbir_bregman_recon run against a given JSON config with Nsight
# Systems (CUDA + OS-runtime tracing), plus a concurrent nvidia-smi GPU
# utilization sample, to help spot CPU-bound vs GPU-bound bottlenecks.
#
# Usage: ./profile_tb86.sh
# Env overrides: BUILD_DIR, BIN, CONFIG, LABEL (defaults to CONFIG's basename)
#
# Output (under ./profiling/<label>_<timestamp>/):
#   tb86.nsys-rep       -- raw Nsight Systems capture (open in the nsys-ui GUI)
#   stdout.log/err.log  -- the program's own stdout/stderr
#   gpu_util.csv        -- nvidia-smi utilization sampled once/sec for the run
#   cuda_gpu_kern_sum.txt    -- time spent per CUDA kernel (GPU-bound work)
#   cuda_api_sum.txt         -- time spent per CUDA API call (sync/memcpy overhead)
#   osrt_sum.txt             -- time spent in OS runtime calls (futex/poll/thread
#                               join etc. -- large numbers here mean CPU-bound
#                               waiting, e.g. host arithmetic or thread joins)
#   cuda_gpu_mem_time_sum.txt / cuda_gpu_mem_size_sum.txt -- H2D/D2H memcpy cost

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${BUILD_DIR:-$REPO_ROOT/build}"
BIN="${BIN:-$BUILD_DIR/mbir_bregman_recon}"
CONFIG="${CONFIG:-$REPO_ROOT/tb86.json}"
LABEL="${LABEL:-$(basename "$CONFIG" .json)}"

if [[ ! -x "$BIN" ]]; then
    echo "error: $BIN not found or not executable -- build it first, e.g.:" >&2
    echo "  cmake --build $BUILD_DIR --target mbir_bregman_recon" >&2
    exit 1
fi
if [[ ! -f "$CONFIG" ]]; then
    echo "error: config $CONFIG not found" >&2
    exit 1
fi
if ! command -v nsys >/dev/null 2>&1; then
    echo "error: nsys (Nsight Systems) not found on PATH" >&2
    exit 1
fi

TS=$(date +%Y%m%d_%H%M%S)
OUT_DIR="$REPO_ROOT/profiling/${LABEL}_${TS}"
mkdir -p "$OUT_DIR"

echo "== profiling $BIN $CONFIG =="
echo "output dir: $OUT_DIR"

# background GPU utilization sampler for the duration of the run
GPU_LOG="$OUT_DIR/gpu_util.csv"
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used \
    --format=csv -l 1 > "$GPU_LOG" 2>&1 &
GPU_SAMPLER_PID=$!
cleanup() { kill "$GPU_SAMPLER_PID" >/dev/null 2>&1 || true; }
trap cleanup EXIT

NSYS_REPORT="$OUT_DIR/$LABEL"
START_TS=$(date +%s)
nsys profile \
    --trace=cuda,osrt \
    --output="$NSYS_REPORT" \
    --force-overwrite=true \
    "$BIN" "$CONFIG" \
    > "$OUT_DIR/stdout.log" 2> "$OUT_DIR/stderr.log"
RUN_STATUS=$?
END_TS=$(date +%s)

cleanup
trap - EXIT

echo "run exited with status $RUN_STATUS after $((END_TS - START_TS))s (see $OUT_DIR/stdout.log)"

if [[ -f "$NSYS_REPORT.nsys-rep" ]]; then
    # --force-export=true: without it, nsys stats reuses/complains about a
    # stale sqlite export across these back-to-back invocations (they can
    # land in the same mtime second) and silently emits an empty report.
    for report in cuda_gpu_kern_sum cuda_api_sum osrt_sum cuda_gpu_mem_time_sum cuda_gpu_mem_size_sum; do
        nsys stats --report "$report" --format column --force-export=true "$NSYS_REPORT.nsys-rep" \
            > "$OUT_DIR/${report}.txt" 2>&1
    done
    echo "nsys summary reports written to $OUT_DIR/*.txt"
else
    echo "warning: $NSYS_REPORT.nsys-rep not found, skipping stats generation" >&2
fi

echo
echo "artifacts in $OUT_DIR:"
ls -la "$OUT_DIR"
