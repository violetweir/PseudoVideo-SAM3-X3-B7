#!/usr/bin/env bash
set -euo pipefail

cd /Data_8TB/lht/PseudoVideo-SAM3-X3-B7
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

PY="${PY:-/home/violet/anaconda3/envs/sam3/bin/python}"
BASE="${BASE:-/Data_8TB/lht/models/modelscope/models/facebook--sam3/snapshots/master/sam3.pt}"
ROUTECO="${ROUTECO:-work/kvasir_1pct_anchors/routeco_sam3_v1/train_routeco_v1/routeco_step000400.pt}"
ROOT="${ROOT:-work/kvasir_1pct_anchors/stage1_feature_knn_b7_validation}"

for mode in \
  anchor_conditioned_target_pooling \
  anchor_conditioned_patch_correspondence
do
  "$PY" scripts/stage1_eval_routes_forward_only.py \
    --checkpoint "$BASE" \
    --routeco-checkpoint "$ROUTECO" \
    --mode "$mode" \
    --root "$ROOT" \
    --split validation \
    --canvas 512 \
    --resume
done
