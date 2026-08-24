#!/usr/bin/env bash
# Evaluate selected LoRA epochs on validation only. Test remains locked.
set -Eeuo pipefail

cd /Data_8TB/lht/PseudoVideo-SAM3-X3-B7

PY=/home/violet/anaconda3/envs/sam3/bin/python
BASE=/Data_8TB/lht/models/modelscope/models/facebook--sam3/snapshots/master/sam3.pt
PHASE=work/rerun_c0_256_sam3knn_s256_base
RUN=$PHASE/medsam3_lora_b0_b6_e50
LORA_DIR=$RUN/lora_weights
EVAL_ROOT=$RUN/checkpoint_eval
ROUTE_SOURCE=$PHASE/stage1_feature_knn_b0_b6
VAL_PRED=$PHASE/predictions/X3_best/student_predictions_validation.jsonl
DEVICE=${DEVICE:-1}
export CUDA_VISIBLE_DEVICES=$DEVICE

MODES=(
  sam3enc_anchor_conditioned_target_pooling
  sam3enc_anchor_conditioned_patch_correspondence
)
NODES=(1 2 3 4 5 6 7 8 9 10 11 12 15 20 25 30 35 40 45 50)

mkdir -p "$EVAL_ROOT/merged"
rm -f "$EVAL_ROOT/VALIDATION_QUEUE_FAILED"
trap 'touch "$EVAL_ROOT/VALIDATION_QUEUE_FAILED"' ERR

summarize_completed() {
  "$PY" scripts/summarize_c0_256_b7_lora_checkpoints.py \
    --eval-root "$EVAL_ROOT" \
    --val-stats "$LORA_DIR/val_stats.json" \
    --output-json "$EVAL_ROOT/validation_checkpoint_summary.json" \
    --output-tsv "$EVAL_ROOT/validation_checkpoint_summary.tsv" \
    --best-epoch-output "$EVAL_ROOT/best_validation_epoch.txt"
}

evaluate_validation_epoch() {
  local epoch=$1 tag lora merged qroot mode
  printf -v tag 'e%02d' "$epoch"
  lora=$LORA_DIR/epoch_${epoch}_lora_weights.pt
  merged=$EVAL_ROOT/merged/${tag}_merged_video.pt
  qroot=$EVAL_ROOT/$tag/quality_root

  while [[ ! -f $lora ]]; do
    if [[ -f $RUN/TRAIN_FAILED ]]; then
      echo "[queue] training failed while waiting for $lora"
      exit 1
    fi
    echo "[queue] $(date '+%F %T') waiting for epoch $epoch"
    sleep 60
  done
  if [[ -f $EVAL_ROOT/$tag/VALIDATION_COMPLETE ]]; then
    summarize_completed
    return
  fi

  echo "[queue] $(date '+%F %T') merge $tag"
  if [[ ! -f $merged ]]; then
    "$PY" scripts/merge_sam3_lora_video_checkpoint.py \
      --base-checkpoint "$BASE" --lora-weights "$lora" --output "$merged"
  fi

  for mode in "${MODES[@]}"; do
    mkdir -p "$qroot/$mode"
    ln -sfn "$(realpath "$ROUTE_SOURCE/$mode/validation_pool0_stage1")" \
      "$qroot/$mode/validation_pool0_stage1"
    echo "[queue] $(date '+%F %T') $tag validation $mode"
    "$PY" scripts/eval_route_propagation_quality.py \
      --checkpoint "$merged" --mode "$mode" --root "$qroot" \
      --split validation --canvas 256 --resume
  done

  "$PY" scripts/build_c0_256_b7_lora_manifest.py \
    --quality-root "$qroot" \
    --student-predictions "$VAL_PRED" \
    --split validation \
    --modes "${MODES[@]}" \
    --output "$EVAL_ROOT/$tag/b7_selected_validation.jsonl" \
    --summary "$EVAL_ROOT/$tag/b7_selected_validation.summary.json" \
    --min-bridge 0 --max-bridge 6 --min-b7 0 --canvas 256
  touch "$EVAL_ROOT/$tag/VALIDATION_COMPLETE"
  summarize_completed
  echo "[queue] $(date '+%F %T') $tag validation complete"
}

for epoch in "${NODES[@]}"; do
  evaluate_validation_epoch "$epoch"
done
summarize_completed
touch "$EVAL_ROOT/VALIDATION_QUEUE_COMPLETE"
echo "[queue] $(date '+%F %T') validation-only queue complete; test not run"
