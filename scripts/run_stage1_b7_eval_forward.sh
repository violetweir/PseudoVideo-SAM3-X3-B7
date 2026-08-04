#!/usr/bin/env bash
set -euo pipefail

cd /Data_8TB/lht/PseudoVideo-SAM3-X3-B7
export CUDA_VISIBLE_DEVICES=0

PY=/home/violet/anaconda3/envs/sam3/bin/python
ROOT=/Data_8TB/lht/PseudoVideo-SAM3-X3-B7/work/kvasir_1pct_anchors/stage1_feature_knn_b7
CKPT=/Data_8TB/lht/models/modelscope/models/facebook--sam3/snapshots/master/sam3.pt

for mode in \
  t18_corrected \
  dino_global_pooling \
  dino_patch_average \
  anchor_conditioned_target_pooling \
  anchor_conditioned_patch_correspondence
do
  echo "===== forward_only_base_no_ft_b7 ${mode} ====="
  "$PY" scripts/stage1_eval_routes_forward_only.py \
    --checkpoint "$CKPT" \
    --mode "$mode" \
    --root "$ROOT" \
    --canvas 512 \
    --resume
done

echo FORWARD_ONLY_BASE_NO_FT_B7_DONE
