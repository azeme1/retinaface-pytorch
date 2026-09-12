#!/bin/bash
# Runs export_check.py --format coreml (PyTorch vs CoreML, real WIDER FACE AP)
# for every CoreML export currently staged on the Hugging Face repo, for every
# backbone except resnet50 (still training). macOS only -- CoreML's
# .predict() needs libcoremlpython, which doesn't exist on Linux.
#
# Which backend to run every checkpoint on is a REQUIRED first argument --
# CPU or MPS, no default, on purpose. CoreML's own default (letting it pick
# ANE/GPU/CPU per-op) has two confirmed GPU/ANE-only numerical bugs on this
# graph: an exp() overflow in box decode (fixed separately, see
# utils/box_utils.py's _MAX_EXP_INPUT clamp) and, even after that fix,
# near-uniform ~0.99 confidence on many checkpoints that collapses AP to
# ~0%% despite otherwise-sane box coordinates. A silent default here would
# just reintroduce the same "trusted a broken backend without realizing it"
# problem export_check.py's own --compute-units already refuses to allow --
# so this script refuses too: you must say CPU or MPS.
#
# CPU maps to coremltools' CPU_ONLY (confirmed correct on every checkpoint
# tested so far). MPS maps to CPU_AND_GPU -- Apple's GPU compute path is
# Metal/MPS under the hood, which is what CPU_AND_GPU actually runs on.
#
# Usage:
#   pip install coremltools opencv-python numpy torch tqdm pillow
#   export HF_TOKEN=hf_...          # required -- azemel/retinaface-xs is private
#   ./validate_coreml.sh CPU|MPS [n_images]
#
# n_images (optional, default: full WIDER FACE val set, 3226 images -- the
# only way to get an AP comparable across runs; pass a smaller number, e.g.
# 200, for a much faster spot check first).
#
# One full log file per (network, level) pair is written to
# ./coreml_verify_logs/ (named ..._CPU.log / ..._MPS.log so both backends'
# runs coexist). A combined coreml_verify_summary_CPU.log or
# coreml_verify_summary_MPS.log is also built per backend as each checkpoint
# finishes: on success it gets the full Easy/Medium/Hard/Average
# PyTorch-vs-CoreML table plus the raw numeric-parity block (box/score/
# landmark max+mean diff before NMS); on failure it gets the last 20 lines
# of that run's log, so most failures are diagnosable from the summary alone.
#
# Run it twice, once with CPU and once with MPS, then diff
# coreml_verify_summary_CPU.log against coreml_verify_summary_MPS.log to see
# exactly which checkpoints hit the GPU/ANE bug -- CPU is the trustworthy
# number either way.
set -uo pipefail

: "${HF_TOKEN:?Set HF_TOKEN to a token with read access to azemel/retinaface-xs}"

if [ $# -lt 1 ] || [ -z "$1" ]; then
  echo "Usage: $0 CPU|MPS [n_images]" >&2
  echo "  backend is REQUIRED, no default -- CPU (coremltools CPU_ONLY) or MPS (coremltools CPU_AND_GPU)" >&2
  exit 1
fi

case "$1" in
  CPU)  COMPUTE_UNITS="CPU_ONLY" ;;
  MPS)  COMPUTE_UNITS="CPU_AND_GPU" ;;
  *)    echo "Invalid backend '$1' -- must be CPU or MPS" >&2; exit 1 ;;
esac

REPO="azemel/retinaface-xs"
IMAGE_SIZE=640
N_IMAGES="${2:-}"
LOG_DIR="./coreml_verify_logs"
SUMMARY="./coreml_verify_summary_${1}.log"

mkdir -p "$LOG_DIR"
: > "$SUMMARY"

# network:levels, matching what's actually staged on the HF repo right now
# (best-mean-AP level, 256, 2, and the selected/bold level per backbone).
declare -a PAIRS=(
  "mobilenetv1:c2 c7 c64 c256"
  "mobilenetv1_0.25:c2 c12 c128 c256"
  "mobilenetv1_0.50:c2 c8 c256"
  "mobilenetv2:c2 c5 c64 c256"
  "resnet18:c2 c5 c32 c256"
  "resnet34:c2 c4 c128 c256"
)

for pair in "${PAIRS[@]}"; do
  network="${pair%%:*}"
  levels="${pair#*:}"
  for level in $levels; do
    name="${network}_${level}"
    log_file="${LOG_DIR}/${name}_${1}.log"
    echo "=== ${name} (${1}) -> ${log_file} ==="
    if [ -n "$N_IMAGES" ]; then
      python3 export_check.py --format coreml \
        --network "$network" \
        --hf-repo "$REPO" --hf-level "$level" \
        --input-color-order rgb \
        --compute-units "$COMPUTE_UNITS" \
        --image-size "$IMAGE_SIZE" \
        --n-images "$N_IMAGES" \
        > "$log_file" 2>&1
    else
      python3 export_check.py --format coreml \
        --network "$network" \
        --hf-repo "$REPO" --hf-level "$level" \
        --input-color-order rgb \
        --compute-units "$COMPUTE_UNITS" \
        --image-size "$IMAGE_SIZE" \
        > "$log_file" 2>&1
    fi
    status=$?

    {
      echo
      echo "########## ${name} (${1}) ##########"
    } >> "$SUMMARY"

    if [ $status -ne 0 ]; then
      echo "FAILED (exit $status) -- last 20 lines of ${log_file}:" >> "$SUMMARY"
      tr '\r' '\n' < "$log_file" | tail -20 >> "$SUMMARY"
      echo "${name}: FAILED (exit $status) -- tail written to ${SUMMARY}, full log at ${log_file}"
      continue
    fi

    # Everything from the "PyTorch vs COREML" comparison header onward: the
    # full Easy/Medium/Hard/Average table (print_comparison_table) AND the
    # numeric-parity block (raw box/score/landmark max+mean diff before NMS)
    # that follows it -- not just the single Average line, per request.
    # export_check.py prints the format name upper-cased (fmt.upper()).
    block=$(tr '\r' '\n' < "$log_file" | awk "/${network//./\\.}: PyTorch vs COREML ===/,0")
    if [ -z "$block" ]; then
      echo "exited 0 but no comparison block found -- check ${log_file} directly" >> "$SUMMARY"
      echo "${name}: exited 0 but no comparison block found -- check ${log_file}"
      continue
    fi
    echo "$block" >> "$SUMMARY"

    avg_line=$(echo "$block" | grep "^Average")
    echo "${name}: OK -- ${avg_line}"
  done
done

echo
echo "All done (backend=${1}). Per-run logs in ${LOG_DIR}/, summary in ${SUMMARY}."
echo "Run this again with the other backend and diff the two summaries to see which checkpoints hit "
echo "the GPU/ANE numerics bug -- CPU is the trustworthy number either way."
