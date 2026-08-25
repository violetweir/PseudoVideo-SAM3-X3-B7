#!/usr/bin/env bash
# Round-2A: freeze the SAM3-base KNN topology and change only propagation to e33.
set -Eeuo pipefail

cd /Data_8TB/lht/PseudoVideo-SAM3-X3-B7

PY=/home/violet/anaconda3/envs/sam3/bin/python
PHASE=work/rerun_c0_256_round2a_fixed_knn_e33
ROUTES=work/rerun_c0_256_sam3knn_s256_base/stage1_feature_knn_b0_b6
E33=work/rerun_c0_256_sam3knn_s256_base/medsam3_lora_b0_b6_e50/e33_full_evaluation
CKPT=$E33/e33_merged_video.pt
QROOT=$PHASE/quality_root
LOG=$PHASE/train_propagation.log
export CUDA_VISIBLE_DEVICES=${DEVICE:-0}

MODES=(
  sam3enc_anchor_conditioned_target_pooling
  sam3enc_anchor_conditioned_patch_correspondence
)

mkdir -p "$PHASE"
rm -f "$PHASE/TRAIN_PROP_FAILED"
trap 'touch "$PHASE/TRAIN_PROP_FAILED"' ERR

for mode in "${MODES[@]}"; do
  mkdir -p "$QROOT/$mode"

  # Route files are symlinked from the frozen SAM3-base KNN topology.
  for split in train validation test; do
    ln -sfn "$(realpath "$ROUTES/$mode/${split}_pool0_stage1")" \
      "$QROOT/$mode/${split}_pool0_stage1"
  done

  # Validation and test were already evaluated with the exact same e33
  # checkpoint and fixed topology. Reuse those immutable results.
  for split in validation test; do
    ln -sfn "$(realpath "$E33/quality_root/$mode/propagation_quality_${split}")" \
      "$QROOT/$mode/propagation_quality_${split}"
  done

  echo "[round2a] $(date '+%F %T') train propagation $mode GPU=${CUDA_VISIBLE_DEVICES}" | tee -a "$LOG"
  "$PY" scripts/eval_route_propagation_quality.py \
    --checkpoint "$CKPT" --mode "$mode" --root "$QROOT" \
    --split train --canvas 256 --resume \
    2>&1 | tee -a "$LOG"
done

for split in train validation test; do
  "$PY" scripts/summarize_c0_256_bridge_metrics.py \
    --quality-root "$QROOT" --split "$split" --modes "${MODES[@]}" \
    --min-bridge 0 --max-bridge 6 \
    --output-json "$PHASE/two_mode_b0_b6_${split}.json" \
    --output-tsv "$PHASE/two_mode_b0_b6_${split}.tsv" \
    2>&1 | tee -a "$LOG"
done

touch "$PHASE/TRAIN_PROP_COMPLETE"
echo "[round2a] $(date '+%F %T') train propagation complete" | tee -a "$LOG"

