#!/usr/bin/env bash
set -euo pipefail

cd /Data_8TB/lht/PseudoVideo-SAM3-X3-B7
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/Data_8TB/lht/PseudoVideo-SAM3-X3-B7/scripts

PY=/home/violet/anaconda3/envs/sam3/bin/python
ROOT=work/kvasir_1pct_anchors/stage1_feature_knn_vitb256
CKPT=work/kvasir_1pct_anchors/video_checkpoints/lora_p491_e20_merged_video.pt
EVAL_NAME=eval_lora_p491_e20

eval_routes () {
  local key=$1
  echo "[eval] $key start $(date '+%F %T')"
  "$PY" scripts/stage1_eval_routes_forward_only.py \
    --checkpoint "$CKPT" \
    --mode "$key" \
    --root "$ROOT" \
    --canvas 256 \
    --eval-name "$EVAL_NAME" \
    --resume
  echo "[eval] $key done $(date '+%F %T')"
}

for key in \
  t18_corrected \
  dino_global_pooling \
  dino_patch_average \
  anchor_conditioned_target_pooling \
  anchor_conditioned_target_pooling__knn_cls \
  anchor_conditioned_target_pooling__knn_pooled \
  anchor_conditioned_target_pooling__knn_cond \
  anchor_conditioned_patch_correspondence \
  anchor_conditioned_patch_correspondence__knn_cls \
  anchor_conditioned_patch_correspondence__knn_pooled \
  anchor_conditioned_patch_correspondence__knn_cond
do
  eval_routes "$key"
done

"$PY" scripts/summarize_stage1_knn_vitb256.py \
  --root "$ROOT" \
  --eval-name "$EVAL_NAME" \
  --label lora_p491_e20 \
  --title "Kvasir test, checkpoint=lora_p491_e20 (LoRA p491 e20), forward-only Dice @ canvas 256, DINOv3 @256"

echo "LORA_P491_E20_DONE"
