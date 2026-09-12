#!/bin/bash
# Runs export_check.py --format tflite (PyTorch vs TFLite, real WIDER FACE AP)
# for every TFLite export currently staged on the Hugging Face repo, for
# every backbone except resnet50 (still training). Works on any platform.
#
# CPU is a REQUIRED first argument -- not because there's a choice (this
# repo wires no GPU delegate for the TFLite interpreter, see export_tflite.py
# -- tf.lite.Interpreter is CPU-only here regardless), but for consistency
# with validate_coreml.sh/validate_onnx.sh's own required-backend argument,
# and so the PyTorch reference side is also explicitly pinned to CPU
# (CUDA_VISIBLE_DEVICES="") rather than silently running on GPU while the
# TFLite side runs on CPU -- a same-device comparison, and one that can't
# contend with something else training on a shared GPU box.
#
# Usage:
#   pip install tensorflow opencv-python numpy torch tqdm
#   export HF_TOKEN=hf_...          # required -- azemel/retinaface-xs is private
#   ./validate_tflite.sh CPU [n_images]
#
# n_images (optional, default: full WIDER FACE val set, 3226 images -- the
# only way to get an AP comparable across runs; pass a smaller number, e.g.
# 200, for a much faster spot check first).
#
# One full log file per (network, level) pair is written to
# ./tflite_verify_logs/. A combined tflite_verify_summary.log is also built
# as each checkpoint finishes: on success it gets the full Easy/Medium/Hard/
# Average PyTorch-vs-TFLite table plus the raw numeric-parity block
# (box/score/landmark max+mean diff before NMS); on failure it gets the last
# 20 lines of that run's log, so most failures are diagnosable from the
# summary alone.
set -uo pipefail

: "${HF_TOKEN:?Set HF_TOKEN to a token with read access to azemel/retinaface-xs}"

if [ $# -lt 1 ] || [ -z "$1" ]; then
  echo "Usage: $0 CPU [n_images]" >&2
  echo "  backend is REQUIRED, no default -- CPU is the only supported value (no GPU delegate wired here)" >&2
  exit 1
fi

if [ "$1" != "CPU" ]; then
  echo "Invalid backend '$1' -- must be CPU (no GPU delegate wired for TFLite in this repo)" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES=""  # pin the PyTorch reference to CPU too -- see header

REPO="azemel/retinaface-xs"
IMAGE_SIZE=640
N_IMAGES="${2:-}"
LOG_DIR="./tflite_verify_logs"
SUMMARY="./tflite_verify_summary.log"

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
    log_file="${LOG_DIR}/${name}.log"
    echo "=== ${name} -> ${log_file} ==="
    if [ -n "$N_IMAGES" ]; then
      python3 export_check.py --format tflite \
        --network "$network" \
        --hf-repo "$REPO" --hf-level "$level" \
        --image-size "$IMAGE_SIZE" \
        --n-images "$N_IMAGES" \
        > "$log_file" 2>&1
    else
      python3 export_check.py --format tflite \
        --network "$network" \
        --hf-repo "$REPO" --hf-level "$level" \
        --image-size "$IMAGE_SIZE" \
        > "$log_file" 2>&1
    fi
    status=$?

    {
      echo
      echo "########## ${name} ##########"
    } >> "$SUMMARY"

    if [ $status -ne 0 ]; then
      echo "FAILED (exit $status) -- last 20 lines of ${log_file}:" >> "$SUMMARY"
      tr '\r' '\n' < "$log_file" | tail -20 >> "$SUMMARY"
      echo "${name}: FAILED (exit $status) -- tail written to ${SUMMARY}, full log at ${log_file}"
      continue
    fi

    # Everything from the "PyTorch vs TFLITE" comparison header onward: the
    # full Easy/Medium/Hard/Average table (print_comparison_table) AND the
    # numeric-parity block (raw box/score/landmark max+mean diff before NMS)
    # that follows it -- not just the single Average line.
    # export_check.py prints the format name upper-cased (fmt.upper()).
    block=$(tr '\r' '\n' < "$log_file" | awk "/${network//./\\.}: PyTorch vs TFLITE ===/,0")
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
echo "All done. Per-run logs in ${LOG_DIR}/, summary in ${SUMMARY}."
