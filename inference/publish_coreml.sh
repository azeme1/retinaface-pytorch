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
# trust here. Publish with BOTH: CPU_ONLY alone cannot see the GPU-path bug
# (dense spatial convs palettized at 2/4/8 bits are wrong on CPU_AND_GPU --
# see export_coreml.py's _GPU_SAFE_SPATIAL_NBITS).
#
# Usage:
#   pip install coremltools opencv-python numpy torch tqdm pillow huggingface_hub
#   export HF_TOKEN=hf_...          # required -- WRITE access to azemel/retinaface-xs for the upload step
#   ./publish_coreml.sh BOTH [match_tolerance_pct]     # validate on CPU_ONLY AND CPU_AND_GPU, upload only if both match
#   ONLY_NETWORKS="resnet50" ./publish_coreml.sh BOTH  # restrict to some backbones
#   ONLY_NETWORKS="mobilenetv1" ONLY_LEVELS="c256" ./publish_coreml.sh BOTH  # ...and some levels
#   ./publish_coreml.sh CPU|MPS [match_tolerance_pct]  # a single compute unit
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
  echo "Usage: [ONLY_NETWORKS=\"resnet50 ...\"] $0 CPU|MPS|BOTH [match_tolerance_pct]" >&2
  echo "  backend is REQUIRED, no default -- CPU (coremltools CPU_ONLY), MPS (coremltools CPU_AND_GPU) or BOTH" >&2
  echo "  (BOTH validates on each and uploads only if every one matches PyTorch -- use this to publish)" >&2
  exit 1
fi

case "$1" in
  CPU)  COMPUTE_UNITS_LIST="CPU_ONLY" ;;
  MPS)  COMPUTE_UNITS_LIST="CPU_AND_GPU" ;;
  BOTH) COMPUTE_UNITS_LIST="CPU_AND_GPU CPU_ONLY" ;;   # GPU (MPS) first: it is the path that can be wrong
  *)    echo "Invalid backend '$1' -- must be CPU, MPS or BOTH" >&2; exit 1 ;;
esac

REPO="azemel/retinaface-xs"
IMAGE_SIZE=640
MATCH_TOLERANCE_PCT="${2:-0.5}"
LOG_DIR="${INFER_DIR}/coreml_publish_logs"
BUILD_DIR="${INFER_DIR}/coreml_publish_build"
SUMMARY="${INFER_DIR}/coreml_publish_summary_${1}.log"

mkdir -p "$LOG_DIR" "$BUILD_DIR"
touch "$SUMMARY"   # append, never truncate: earlier runs' results stay in the summary

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
  "resnet50:c2 c4 c128 c256"
)

for pair in "${PAIRS[@]}"; do
  network="${pair%%:*}"
  levels="${pair#*:}"
  if [ -n "${ONLY_NETWORKS:-}" ] && [[ " ${ONLY_NETWORKS} " != *" ${network} "* ]]; then
    continue
  fi
  for level in $levels; do
    if [ -n "${ONLY_LEVELS:-}" ] && [[ " ${ONLY_LEVELS} " != *" ${level} "* ]]; then
      continue
    fi
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
    ckpt_url="https://huggingface.co/${REPO}/resolve/main/checkpoints/${network}/pytorch/${network}_${level}.zip"

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

    {
      echo
      echo "########## ${name} (${1}) ##########"
    } >> "$SUMMARY"

    all_match=1
    for unit in $COMPUTE_UNITS_LIST; do
      unit_log="${LOG_DIR}/${name}_${unit}.log"
      echo "########## ${name}: full 3226-image WIDER FACE validation (pytorch vs NEW coreml, ${unit}) -> ${unit_log} ##########" >> "$log_file"
      python3 "${INFER_DIR}/export_check.py" --format coreml --network "$network" \
        --checkpoint-url "$ckpt_url" --hf-token "$HF_TOKEN" --artifact "$mlpackage" \
        --input-color-order rgb --compute-units "$unit" \
        --image-size "$IMAGE_SIZE" > "$unit_log" 2>&1
      check_status=$?
      if [ $check_status -ne 0 ]; then
        echo "${name} [${unit}]: FAILED (exit $check_status) -- last 20 lines of ${unit_log}:" >> "$SUMMARY"
        tr '\r' '\n' < "$unit_log" | tail -20 >> "$SUMMARY"
        echo "${name} [${unit}]: FAILED (exit $check_status) -- see ${unit_log}"
        all_match=0
        continue
      fi

      # Same block-extraction convention as validate_coreml.sh.
      block=$(tr '\r' '\n' < "$unit_log" | awk "/${network//./\\.}: PyTorch vs COREML ===/,0")
      if [ -z "$block" ]; then
        echo "${name} [${unit}]: exited 0 but no comparison block found -- check ${unit_log}" | tee -a "$SUMMARY"
        all_match=0
        continue
      fi
      { echo "[${unit}]"; echo "$block"; } >> "$SUMMARY"

      # Parsed from the comparison table itself ("Average  60.77%  60.77%  +0.00%"),
      # not by searching near the individual report headers: those sit tens of
      # lines before their Average line once tqdm's \r progress is expanded.
      avg_line=$(echo "$block" | grep "^Average")
      pt_mean=$(echo "$avg_line" | awk '{print $2}' | tr -d '%')
      cm_mean=$(echo "$avg_line" | awk '{print $3}' | tr -d '%')
      if [ -z "$pt_mean" ] || [ -z "$cm_mean" ]; then
        echo "${name} [${unit}]: could not parse Average AP -- check ${unit_log} manually" | tee -a "$SUMMARY"
        all_match=0
        continue
      fi

      is_match=$(python3 -c "print(1 if abs($pt_mean - $cm_mean) <= $MATCH_TOLERANCE_PCT else 0)")
      diff=$(python3 -c "print(f'{abs($pt_mean - $cm_mean):.3f}')")
      echo "${name} [${unit}]: pytorch=${pt_mean}% coreml=${cm_mean}% diff=${diff}pp (tolerance ${MATCH_TOLERANCE_PCT}pp)" | tee -a "$SUMMARY"
      [ "$is_match" = "1" ] || all_match=0
    done

    if [ "$all_match" != "1" ]; then
      echo "${name}: NOT uploading -- did not match PyTorch on every requested compute unit (${COMPUTE_UNITS_LIST})" | tee -a "$SUMMARY"
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
    path_in_repo='checkpoints/$network/coreml/${network}_${level}.zip',
    repo_id='$REPO',
    repo_type='model',
    commit_message='Fix CoreML export for $network $level: unique-mode palettization + RGB permute (was silently near-0%% AP)',
)
" >> "$log_file" 2>&1
    upload_status=$?
    if [ $upload_status -ne 0 ]; then
      echo "${name}: UPLOAD FAILED (exit ${upload_status}) -- see ${log_file}" | tee -a "$SUMMARY"
    else
      echo "${name}: uploaded -- checkpoints/${network}/coreml/${network}_${level}.zip" | tee -a "$SUMMARY"
    fi
  done
done

echo
echo "All done (backend=${1}). Per-run logs in ${LOG_DIR}/, summary in ${SUMMARY}, built .mlpackages in ${BUILD_DIR}/."
echo "Run ./validate_coreml.sh ${1} now to confirm the newly-uploaded artifacts check out from HF."
