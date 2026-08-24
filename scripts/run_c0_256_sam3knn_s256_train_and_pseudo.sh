#!/usr/bin/env bash
# Continue with train propagation and b0-b6 pseudo-label preparation.
set -euo pipefail

cd /Data_8TB/lht/PseudoVideo-SAM3-X3-B7

PY=/home/violet/anaconda3/envs/sam3/bin/python
MKPY=/home/violet/anaconda3/envs/mkunet_mamba/bin/python
CKPT=/Data_8TB/lht/models/modelscope/models/facebook--sam3/snapshots/master/sam3.pt
PHASE=work/rerun_c0_256_sam3knn_s256_base
ROOT="$PHASE/stage1_feature_knn_b0_b6"
LOG="$PHASE/train_propagation_and_pseudo.log"
export CUDA_VISIBLE_DEVICES=1

TARGET_MODE=sam3enc_anchor_conditioned_target_pooling
PATCH_MODE=sam3enc_anchor_conditioned_patch_correspondence
MODES=("$TARGET_MODE" "$PATCH_MODE")

while [[ ! -f "$PHASE/VALIDATION_COMPLETE" ]]; do
  echo "[train-prop] $(date '+%F %T') waiting for validation" | tee -a "$LOG"
  sleep 60
done

echo "[train-prop] $(date '+%F %T') start $PATCH_MODE on GPU1" | tee -a "$LOG"
"$PY" scripts/eval_route_propagation_quality.py \
  --checkpoint "$CKPT" \
  --mode "$PATCH_MODE" \
  --root "$ROOT" \
  --split train \
  --canvas 256 \
  --resume \
  >> "$LOG" 2>&1
touch "$PHASE/PATCH_TRAIN_PROP_COMPLETE"
echo "[train-prop] $(date '+%F %T') complete $PATCH_MODE" | tee -a "$LOG"

while [[ ! -f "$PHASE/TARGET_TRAIN_PROP_COMPLETE" ]]; do
  echo "[train-prop] $(date '+%F %T') waiting for GPU0 target-pooling train propagation" | tee -a "$LOG"
  sleep 60
done

"$PY" scripts/summarize_c0_256_bridge_metrics.py \
  --quality-root "$ROOT" \
  --split train \
  --modes "${MODES[@]}" \
  --min-bridge 0 --max-bridge 6 \
  --output-json "$PHASE/train_bridge_b0_b6.json" \
  --output-tsv "$PHASE/train_bridge_b0_b6.tsv" \
  >> "$LOG" 2>&1

# Main b0-b6 pool.
"$MKPY" scripts/select_phase1_mainline_pseudo568.py \
  --quality-root "$ROOT" \
  --min-bridge 0 --max-bridge 6 \
  --output "$PHASE/pseudo_manifest_original_b0_b6.jsonl" \
  >> "$LOG" 2>&1
"$MKPY" scripts/prepare_phase1_s3_consensus.py \
  --original-manifest "$PHASE/pseudo_manifest_original_b0_b6.jsonl" \
  --quality-root "$ROOT" \
  --min-bridge 0 --max-bridge 6 \
  --output-root "$PHASE/S3_consensus_b0_b6" \
  >> "$LOG" 2>&1

# Frozen b3-b6 control from the same propagation output.
"$MKPY" scripts/select_phase1_mainline_pseudo568.py \
  --quality-root "$ROOT" \
  --min-bridge 3 --max-bridge 6 \
  --output "$PHASE/pseudo_manifest_original_b3_b6_control.jsonl" \
  >> "$LOG" 2>&1

touch "$PHASE/TRAIN_PROP_AND_PSEUDO_COMPLETE"
echo "[train-prop] $(date '+%F %T') train propagation and pseudo preparation complete" | tee -a "$LOG"
