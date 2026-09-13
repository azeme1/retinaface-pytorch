#!/bin/bash
# Rebuilds the CoreML "prior + decoder fusion" export (RetinaStaticExportWrapper,
# Stage 2 in export_common.py -- backbone + fused priors/decode) for every
# (network, level) pair currently staged on the Hugging Face repo, validates
# each fresh build against the full 3226-image WIDER FACE val set, and
# uploads it to replace the existing HF-hosted .mlpackage ONLY when the new
# build's CoreML AP matches its PyTorch reference within MATCH_TOLERANCE_PCT.
#
# Why this needs to exist at all: the .mlpackage files currently on
# azemel/retinaface-xs predate two fixes made in this session --
#   1. apply_palette_selective (export_coreml.py) used mode="kmeans", which
#      silently corrupts output whenever num_clusters isn't an exact
#      2**nbits (confirmed: c7, c8, c5, c12, c32, c128 all collapsed to ~0%
#      AP across every backbone) -- fixed to mode="unique" (the weights are
#      already clustered by this project's own training-time quantization,
#      so re-deriving a codebook via kmeans was both redundant and buggy).
#   2. the wrapper's RGB/BGR permute bug documented in export_coreml.py's
#      own comment ("every .mlpackage built before this fix needs
#      re-exporting").
# validate_coreml.sh (this directory) only ever downloads and CHECKS
# whatever's already on HF -- it can't fix a broken artifact, only report
# it as broken. This script is the other half: rebuild, validate, and (if
# it passes) publish the fix. Run validate_coreml.sh afterward to confirm
# the newly-uploaded artifacts now check out.
#
# Same backend-argument requirement as validate_coreml.sh, and for the same
# reason -- see that script's own header comment on why CoreML's own
# default (letting it pick ANE/GPU/CPU per-op) is not safe to silently
# trust here.
#
# Usage:
#   pip install coremltools opencv-python numpy torch tqdm pillow huggingface_hub
#   export HF_TOKEN=hf_...          # required -- WRITE access to azemel/retinaface-xs for the upload step
#   ./publish_coreml.sh CPU|MPS [match_tolerance_pct]
#
# match_tolerance_pct (optional, default 0.5): max acceptable
# |pytorch_mean - coreml_mean| in AP percentage points to call a level a
# match and upload it. This project's own bar for ONNX (see the HF repo's
# README.md) is 0.09%; kept looser here by default since CoreML is a
# different runtime -- tighten if you want ONNX-level strictness.
#
# One full log per (network, level) pair is written to ./coreml_publish_logs/,
# plus a combined coreml_publish_summary_<BACKEND>.log as each finishes,
# same convention as validate_coreml.sh's own logging.
#
# Safe to re-run: skips a (network, level) pair whose log already exists,
# same idempotency convention as this project's other batch scripts
# (evaluate/run_parallel.py's own --force-less default).
set -uo pipefail

# Resolved from the script's own location, not the caller's cwd -- run
# this from the repo root, from inference/, or anywhere else and it lands
# in the same place either way.
INFER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

: "${HF_TOKEN:?Set HF_TOKEN to a token with WRITE access to azemel/retinaface-xs}"

if [ $# -lt 1 ] || [ -z "$1" ]; then
  echo "Usage: $0 CPU|MPS [match_tolerance_pct]" >&2
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
MATCH_TOLERANCE_PCT="${2:-0.5}"
LOG_DIR="${INFER_DIR}/coreml_publish_logs"
BUILD_DIR="${INFER_DIR}/coreml_publish_build"
SUMMARY="${INFER_DIR}/coreml_publish_summary_${1}.log"

mkdir -p "$LOG_DIR" "$BUILD_DIR"
: > "$SUMMARY"

# Same pairs, same source of truth as validate_coreml.sh -- the README's
# "exported cluster counts (CoreML)" table. Don't expand to every pytorch
# level (most have no CoreML export to replace at all).
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
    mlpackage_dir="${BUILD_DIR}/${network}"

    if [ -f "$log_file" ]; then
      echo "${name}: already processed (log exists) -- skipping. Delete ${log_file} to redo."
      continue
    fi

    echo "=== ${name} (${1}) -> ${log_file} ==="
    : > "$log_file"

    # export_coreml.py/export_check.py both take --checkpoint-url directly
    # (see either script's own docstring) and resolve everything else
    # (layers/models/utils, export_common) themselves via their own
    # __file__-anchored sys.path setup -- no separate download step, no
    # path-fixing on this script's side at all.
    ckpt_url="https://huggingface.co/${REPO}/resolve/main/results/${network}/pytorch/${network}_${level}.zip"

    echo "########## ${name}: building corrected CoreML export (Stage 2: backbone + fused priors/decode) ##########" >> "$log_file"
    python3 "${INFER_DIR}/export_coreml.py" --network "$network" \
      --checkpoint-url "$ckpt_url" --hf-token "$HF_TOKEN" \
      --image-size "$IMAGE_SIZE" --out-dir "$mlpackage_dir" >> "$log_file" 2>&1
    build_status=$?
    mlpackage="${mlpackage_dir}/${network}_${level}.mlpackage"
    if [ $build_status -ne 0 ] || [ ! -e "$mlpackage" ]; then
      echo "${name}: BUILD FAILED (exit ${build_status}) -- see ${log_file}" | tee -a "$SUMMARY"
      continue
    fi

    echo "########## ${name}: full 3226-image WIDER FACE validation (pytorch vs NEW coreml) ##########" >> "$log_file"
    python3 "${INFER_DIR}/export_check.py" --format coreml --network "$network" \
      --checkpoint-url "$ckpt_url" --hf-token "$HF_TOKEN" --artifact "$mlpackage" \
      --input-color-order rgb --compute-units "$COMPUTE_UNITS" \
      --image-size "$IMAGE_SIZE" >> "$log_file" 2>&1
    check_status=$?

    {
      echo
      echo "########## ${name} (${1}) ##########"
    } >> "$SUMMARY"

    if [ $check_status -ne 0 ]; then
      echo "FAILED (exit $check_status) -- last 20 lines of ${log_file}:" >> "$SUMMARY"
      tr '\r' '\n' < "$log_file" | tail -20 >> "$SUMMARY"
      echo "${name}: FAILED (exit $check_status) -- tail written to ${SUMMARY}, full log at ${log_file}"
      continue
    fi

    # Same block-extraction convention as validate_coreml.sh.
    block=$(tr '\r' '\n' < "$log_file" | awk "/${network//./\\.}: PyTorch vs COREML ===/,0")
    if [ -z "$block" ]; then
      echo "exited 0 but no comparison block found -- check ${log_file} directly" >> "$SUMMARY"
      echo "${name}: exited 0 but no comparison block found -- check ${log_file}"
      continue
    fi
    echo "$block" >> "$SUMMARY"

    pt_mean=$(tr '\r' '\n' < "$log_file" | grep -A4 "PyTorch (fixed-size wrapper)" | grep "Average:" | head -1 | grep -oE '[0-9.]+')
    cm_mean=$(tr '\r' '\n' < "$log_file" | grep -A4 "^=== ${network}: COREML" | grep "Average:" | head -1 | grep -oE '[0-9.]+')

    if [ -z "$pt_mean" ] || [ -z "$cm_mean" ]; then
      echo "${name}: could not parse Average AP -- check ${log_file} manually, NOT uploading" | tee -a "$SUMMARY"
      continue
    fi

    is_match=$(python3 -c "print(1 if abs($pt_mean - $cm_mean) <= $MATCH_TOLERANCE_PCT else 0)")
    diff=$(python3 -c "print(f'{abs($pt_mean - $cm_mean):.3f}')")
    echo "${name}: pytorch=${pt_mean}% coreml=${cm_mean}% diff=${diff}pp (tolerance ${MATCH_TOLERANCE_PCT}pp)" | tee -a "$SUMMARY"

    if [ "$is_match" != "1" ]; then
      echo "${name}: MISMATCH -- NOT uploading, needs investigation" | tee -a "$SUMMARY"
      continue
    fi

    echo "${name}: MATCH -- uploading corrected .mlpackage to ${REPO}" | tee -a "$SUMMARY"
    stage_dir=$(mktemp -d)
    cp -R "$mlpackage" "${stage_dir}/model.mlpackage"
    upload_zip="${BUILD_DIR}/${name}_upload.zip"
    (cd "$stage_dir" && zip -r -X "$upload_zip" "model.mlpackage" -x '.*') >> "$log_file" 2>&1
    rm -rf "$stage_dir"

    python3 -c "
from huggingface_hub import HfApi
api = HfApi(token='$HF_TOKEN')
api.upload_file(
    path_or_fileobj='$upload_zip',
    path_in_repo='results/$network/coreml/${network}_${level}.zip',
    repo_id='$REPO',
    repo_type='model',
    commit_message='Fix CoreML export for $network $level: unique-mode palettization + RGB permute (was silently near-0%% AP)',
)
" >> "$log_file" 2>&1
    upload_status=$?
    if [ $upload_status -ne 0 ]; then
      echo "${name}: UPLOAD FAILED (exit ${upload_status}) -- see ${log_file}" | tee -a "$SUMMARY"
    else
      echo "${name}: uploaded -- results/${network}/coreml/${network}_${level}.zip" | tee -a "$SUMMARY"
    fi
  done
done

echo
echo "All done (backend=${1}). Per-run logs in ${LOG_DIR}/, summary in ${SUMMARY}, built .mlpackages in ${BUILD_DIR}/."
echo "Run ./validate_coreml.sh ${1} now to confirm the newly-uploaded artifacts check out from HF."
