#!/usr/bin/env bash
set -euo pipefail

cd /Data_8TB/lht/PseudoVideo-SAM3-X3-B7
export CUDA_VISIBLE_DEVICES=0

PY=/home/violet/anaconda3/envs/sam3/bin/python
ROOT=work/kvasir_1pct_anchors/stage1_feature_knn_b7_validation
CKPT=/Data_8TB/lht/models/modelscope/models/facebook--sam3/snapshots/master/sam3.pt

for mode in \
  t18_corrected \
  dino_global_pooling \
  dino_patch_average \
  anchor_conditioned_target_pooling \
  anchor_conditioned_patch_correspondence
do
  echo "===== validation_forward_b7 ${mode} $(date -Is) ====="
  "$PY" scripts/stage1_eval_routes_forward_only.py \
    --checkpoint "$CKPT" \
    --mode "$mode" \
    --root "$ROOT" \
    --split validation \
    --canvas 512 \
    --resume
done

echo "VALIDATION_FORWARD_B7_DONE $(date -Is)"
