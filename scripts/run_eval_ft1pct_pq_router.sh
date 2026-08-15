#!/usr/bin/env bash
set -euo pipefail

cd /Data_8TB/lht/PseudoVideo-SAM3-X3-B7

PY=/home/violet/anaconda3/envs/sam3/bin/python
CKPT=work/kvasir_1pct_anchors/video_checkpoints/ft_1pct_merged_video.pt
ROOT=work/kvasir_1pct_anchors/stage1_feature_knn_b7_ft1pct
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

for mode in \
  anchor_conditioned_target_pooling \
  anchor_conditioned_patch_correspondence
do
  echo "===== propagation quality validation $mode ====="
  "$PY" scripts/eval_route_propagation_quality.py \
    --checkpoint "$CKPT" \
    --mode "$mode" \
    --root "$ROOT" \
    --split validation \
    --canvas 512 \
    --resume
done

for mode in \
  anchor_conditioned_target_pooling \
  anchor_conditioned_patch_correspondence
do
  echo "===== propagation quality test $mode ====="
  "$PY" scripts/eval_route_propagation_quality.py \
    --checkpoint "$CKPT" \
    --mode "$mode" \
    --root "$ROOT" \
    --split test \
    --canvas 512 \
    --resume
done

echo "===== ridge router summary ====="
"$PY" scripts/eval_ft1pct_pq_router.py \
  --root "$ROOT" \
  --min-bridge 3 \
  --max-bridge 6 \
  --output work/kvasir_1pct_anchors/propagation_quality_router_ft1pct_b3_b6_check/summary.json

echo "===== ridge router summary b0-b6 (reference) ====="
"$PY" scripts/eval_ft1pct_pq_router.py \
  --root "$ROOT" \
  --min-bridge 0 \
  --max-bridge 6 \
  --output work/kvasir_1pct_anchors/propagation_quality_router_ft1pct_b0_b6_check/summary.json

echo "[done] ft1pct propagation-quality router finished"
