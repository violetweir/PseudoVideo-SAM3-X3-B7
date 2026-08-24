#!/usr/bin/env bash
# Run target-pooling train propagation independently on physical GPU0.
set -euo pipefail

cd /Data_8TB/lht/PseudoVideo-SAM3-X3-B7

PY=/home/violet/anaconda3/envs/sam3/bin/python
CKPT=/Data_8TB/lht/models/modelscope/models/facebook--sam3/snapshots/master/sam3.pt
PHASE=work/rerun_c0_256_sam3knn_s256_base
ROOT="$PHASE/stage1_feature_knn_b0_b6"
LOG="$PHASE/train_target_gpu0.log"
MODE=sam3enc_anchor_conditioned_target_pooling
export CUDA_VISIBLE_DEVICES=0

while [[ ! -f "$PHASE/ROUTES_COMPLETE" ]]; do
  sleep 30
done

echo "[target-gpu0] $(date '+%F %T') start $MODE" | tee -a "$LOG"
"$PY" scripts/eval_route_propagation_quality.py \
  --checkpoint "$CKPT" \
  --mode "$MODE" \
  --root "$ROOT" \
  --split train \
  --canvas 256 \
  --resume \
  >> "$LOG" 2>&1
touch "$PHASE/TARGET_TRAIN_PROP_COMPLETE"
echo "[target-gpu0] $(date '+%F %T') complete $MODE" | tee -a "$LOG"
