#!/usr/bin/env bash
set -euo pipefail
cd /Data_8TB/lht/PseudoVideo-SAM3-X3-B7
PY=/home/violet/anaconda3/envs/sam3/bin/python
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
CKPT=/Data_8TB/lht/models/modelscope/models/facebook--sam3/snapshots/master/sam3.pt
for mode in \
  t18_corrected \
  dino_global_pooling \
  dino_patch_average \
  anchor_conditioned_target_pooling \
  anchor_conditioned_patch_correspondence
do
  echo "===== eval base_no_ft ${mode} ====="
  "${PY}" scripts/stage1_eval_feature_knn_routes.py \
    --checkpoint "${CKPT}" \
    --model-name base_no_ft \
    --mode "${mode}" \
    --resume
done
