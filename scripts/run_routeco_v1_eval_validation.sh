#!/usr/bin/env bash
set -euo pipefail

cd /Data_8TB/lht/PseudoVideo-SAM3-X3-B7
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

PY="${PY:-/home/violet/anaconda3/envs/sam3/bin/python}"
BASE="${BASE:-/Data_8TB/lht/models/modelscope/models/facebook--sam3/snapshots/master/sam3.pt}"
ROUTECO="${ROUTECO:-work/kvasir_1pct_anchors/routeco_sam3_v1/train_routeco_v1/routeco_step000400.pt}"
ROOT="${ROOT:-work/kvasir_1pct_anchors/stage1_feature_knn_b7_validation}"
OUT_NAME="${OUT_NAME:-eval_routeco_core_v1_step400}"

for mode in \
  anchor_conditioned_target_pooling \
  anchor_conditioned_patch_correspondence
do
  "$PY" scripts/eval_routeco_core_routes.py \
    --base-checkpoint "$BASE" \
    --routeco-checkpoint "$ROUTECO" \
    --root "$ROOT" \
    --mode "$mode" \
    --split validation \
    --output-name "$OUT_NAME" \
    --image-size 1008 \
    --resume
done
