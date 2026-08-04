#!/usr/bin/env bash
set -euo pipefail

cd /Data_8TB/lht/PseudoVideo-SAM3-X3-B7
PY=/home/violet/anaconda3/envs/sam3/bin/python
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

CKPT=work/kvasir_1pct_anchors/video_checkpoints/ft_20pct_latest_after_crash_merged_video.pt

for mode in \
  t18_corrected \
  dino_global_pooling \
  dino_patch_average \
  anchor_conditioned_target_pooling \
  anchor_conditioned_patch_correspondence
do
  echo "===== eval ${mode} ====="
  "${PY}" scripts/stage1_eval_feature_knn_routes.py \
    --checkpoint "${CKPT}" \
    --model-name ft20_latest \
    --mode "${mode}" \
    --resume
done
