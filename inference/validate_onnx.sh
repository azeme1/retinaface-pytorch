#!/bin/bash
# Runs export_check.py --format onnx (PyTorch vs ONNX Runtime, real WIDER FACE
# AP) for every ONNX export currently staged on the Hugging Face repo, for
# every backbone except resnet50 (still training). Works on any platform --
# ONNX Runtime has no macOS-only restriction the way CoreML's .predict() does.
#
# Which execution provider to run every checkpoint on is a REQUIRED first
# argument -- CPU or CUDA, no default, on purpose (matching validate_coreml.sh's
# CPU/MPS choice). No correctness bug has been found forcing this the way
# CoreML's compute-units choice is forced, but auto-selecting CUDA silently
# would risk contending with something else training on a shared GPU box, and
# forcing it also pins the PyTorch reference side onto the SAME device (see
# export_check.py's --onnx-provider --help), so this is a genuine same-device
# comparison rather than ONNX-on-CPU vs a GPU-computed reference.
#
# Usage:
#   pip install onnxruntime-gpu opencv-python numpy torch tqdm   # or plain onnxruntime for CPU only
#   export HF_TOKEN=hf_...          # required -- azemel/retinaface-xs is private
#   ./validate_onnx.sh CPU|CUDA [n_images]
#
# n_images (optional, default: full WIDER FACE val set, 3226 images -- the
# only way to get an AP comparable across runs; pass a smaller number, e.g.
# 200, for a much faster spot check first).
#
# One full log file per (network, level) pair is written to
# ./onnx_verify_logs/ (named ..._CPU.log / ..._CUDA.log so both providers'
# runs coexist). A combined onnx_verify_summary_CPU.log or
# onnx_verify_summary_CUDA.log is also built per provider as each checkpoint
# finishes: on success it gets the full Easy/Medium/Hard/Average
# PyTorch-vs-ONNX table plus the raw numeric-parity block (box/score/
# landmark max+mean diff before NMS); on failure it gets the last 20 lines
# of that run's log, so most failures are diagnosable from the summary alone.
set -uo pipefail

: "${HF_TOKEN:?Set HF_TOKEN to a token with read access to azemel/retinaface-xs}"

if [ $# -lt 1 ] || [ -z "$1" ]; then
  echo "Usage: $0 CPU|CUDA [n_images]" >&2
  echo "  provider is REQUIRED, no default -- CPU or CUDA" >&2
  exit 1
fi

case "$1" in
  CPU|CUDA) ;;
  *) echo "Invalid provider '$1' -- must be CPU or CUDA" >&2; exit 1 ;;
esac

REPO="azemel/retinaface-xs"
IMAGE_SIZE=640
N_IMAGES="${2:-}"
LOG_DIR="./onnx_verify_logs"
SUMMARY="./onnx_verify_summary_${1}.log"

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
      python3 export_check.py --format onnx \
        --network "$network" \
        --hf-repo "$REPO" --hf-level "$level" \
        --onnx-provider "$1" \
        --image-size "$IMAGE_SIZE" \
        --n-images "$N_IMAGES" \
        > "$log_file" 2>&1
    else
      python3 export_check.py --format onnx \
        --network "$network" \
        --hf-repo "$REPO" --hf-level "$level" \
        --onnx-provider "$1" \
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

    # Everything from the "PyTorch vs ONNX" comparison header onward: the
    # full Easy/Medium/Hard/Average table (print_comparison_table) AND the
    # numeric-parity block (raw box/score/landmark max+mean diff before NMS)
    # that follows it -- not just the single Average line.
    # export_check.py prints the format name upper-cased (fmt.upper()).
    block=$(tr '\r' '\n' < "$log_file" | awk "/${network//./\\.}: PyTorch vs ONNX ===/,0")
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
echo "All done (provider=${1}). Per-run logs in ${LOG_DIR}/, summary in ${SUMMARY}."
