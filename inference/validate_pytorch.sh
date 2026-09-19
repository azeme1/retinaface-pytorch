#!/bin/bash
# Real full-val WIDER FACE AP of each published PyTorch checkpoint through the
# fixed-size RetinaStaticExportWrapper (export_check.py --format pytorch, no
# converted artifact): ./validate_pytorch.sh <network> <level>...
# Prediction folders are removed after each run (disk is tight); logs stay in
# ./pytorch_verify_logs/.
set -uo pipefail
: "${LLWLL_ROOT:?}"
net="$1"; shift
mkdir -p pytorch_verify_logs
for lvl in "$@"; do
  python3 export_check.py --format pytorch --network "$net" --hf-repo azemel/retinaface-xs \
    --hf-level "$lvl" --image-size 640 > "pytorch_verify_logs/${net}_${lvl}.log" 2>&1
  echo "${net}_${lvl}: exit $?"
  rm -rf "../results/${net}/pytorch_check_pred_pytorch_${lvl}"
done
