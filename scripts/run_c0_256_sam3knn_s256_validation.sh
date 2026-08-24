#!/usr/bin/env bash
# Run validation first; test remains held out until the downstream protocol is locked.
set -euo pipefail

cd /Data_8TB/lht/PseudoVideo-SAM3-X3-B7

PY=/home/violet/anaconda3/envs/sam3/bin/python
CKPT=/Data_8TB/lht/models/modelscope/models/facebook--sam3/snapshots/master/sam3.pt
PHASE=work/rerun_c0_256_sam3knn_s256_base
ROOT="$PHASE/stage1_feature_knn_b0_b6"
LOG="$PHASE/validation_propagation.log"
export CUDA_VISIBLE_DEVICES=1

MODES=(
  sam3enc_anchor_conditioned_target_pooling
  sam3enc_anchor_conditioned_patch_correspondence
)

while [[ ! -f "$PHASE/ROUTES_COMPLETE" ]]; do
  echo "[validation] $(date '+%F %T') waiting for routes" | tee -a "$LOG"
  sleep 30
done

for mode in "${MODES[@]}"; do
  echo "[validation] $(date '+%F %T') start $mode" | tee -a "$LOG"
  "$PY" scripts/eval_route_propagation_quality.py \
    --checkpoint "$CKPT" \
    --mode "$mode" \
    --root "$ROOT" \
    --split validation \
    --canvas 256 \
    --resume \
    >> "$LOG" 2>&1
  echo "[validation] $(date '+%F %T') complete $mode" | tee -a "$LOG"
done

"$PY" scripts/summarize_c0_256_bridge_metrics.py \
  --quality-root "$ROOT" \
  --split validation \
  --modes "${MODES[@]}" \
  --min-bridge 0 --max-bridge 6 \
  --output-json "$PHASE/validation_bridge_b0_b6.json" \
  --output-tsv "$PHASE/validation_bridge_b0_b6.tsv" \
  >> "$LOG" 2>&1

touch "$PHASE/VALIDATION_COMPLETE"
echo "[validation] $(date '+%F %T') all validation work complete" | tee -a "$LOG"
