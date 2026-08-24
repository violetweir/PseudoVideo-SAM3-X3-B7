#!/usr/bin/env bash
# Evaluate late LoRA checkpoints on physical GPU 0 while the main queue uses GPU 1.
set -euo pipefail

cd /Data_8TB/lht/PseudoVideo-SAM3-X3-B7

PY=/home/violet/anaconda3/envs/sam3/bin/python
BASE=/Data_8TB/lht/models/modelscope/models/facebook--sam3/snapshots/master/sam3.pt
RUN=work/rerun_c0_256_base/medsam3_lora_b7_e50
LORA_DIR="$RUN/lora_weights"
EVAL_ROOT="$RUN/checkpoint_eval"
FILTERED_ROUTES="$EVAL_ROOT/routes_b3_b6"
VAL_PRED=work/rerun_c0_256_base/predictions/X3_best/student_predictions_validation.jsonl
export CUDA_VISIBLE_DEVICES=0

MODES=(
  anchor_conditioned_target_pooling
  anchor_conditioned_patch_correspondence
)

# These are far enough ahead of the GPU-1 queue to avoid concurrent writes.
NODES=(40 50)

summarize_completed() {
  "$PY" scripts/summarize_c0_256_b7_lora_checkpoints.py \
    --eval-root "$EVAL_ROOT" \
    --val-stats "$LORA_DIR/val_stats.json" \
    --output-json "$EVAL_ROOT/validation_checkpoint_summary.json" \
    --output-tsv "$EVAL_ROOT/validation_checkpoint_summary.tsv" \
    --best-epoch-output "$EVAL_ROOT/best_validation_epoch.txt"
}

evaluate_epoch() {
  local epoch=$1 tag lora merged qroot mode
  printf -v tag 'e%02d' "$epoch"
  lora="$LORA_DIR/epoch_${epoch}_lora_weights.pt"
  merged="$EVAL_ROOT/merged/${tag}_merged_video.pt"
  qroot="$EVAL_ROOT/$tag/quality_root"

  if [[ -f "$EVAL_ROOT/$tag/VALIDATION_COMPLETE" ]]; then
    echo "[gpu0-eval] $(date '+%F %T') $tag already complete"
    return
  fi
  if [[ ! -f "$lora" ]]; then
    echo "[gpu0-eval] missing checkpoint: $lora" >&2
    return 1
  fi

  echo "[gpu0-eval] $(date '+%F %T') merge $tag"
  if [[ ! -f "$merged" ]]; then
    "$PY" scripts/merge_sam3_lora_video_checkpoint.py \
      --base-checkpoint "$BASE" \
      --lora-weights "$lora" \
      --output "$merged"
  fi

  for mode in "${MODES[@]}"; do
    mkdir -p "$qroot/$mode"
    ln -sfn "$(realpath "$FILTERED_ROUTES/$mode/validation_pool0_stage1")" \
      "$qroot/$mode/validation_pool0_stage1"
    echo "[gpu0-eval] $(date '+%F %T') $tag validation $mode"
    "$PY" scripts/eval_route_propagation_quality.py \
      --checkpoint "$merged" \
      --mode "$mode" \
      --root "$qroot" \
      --split validation \
      --canvas 256 \
      --resume
  done

  "$PY" scripts/build_c0_256_b7_lora_manifest.py \
    --quality-root "$qroot" \
    --student-predictions "$VAL_PRED" \
    --split validation \
    --output "$EVAL_ROOT/$tag/b7_selected_validation.jsonl"
  touch "$EVAL_ROOT/$tag/VALIDATION_COMPLETE"
  summarize_completed
  echo "[gpu0-eval] $(date '+%F %T') $tag validation complete"
}

for epoch in "${NODES[@]}"; do
  evaluate_epoch "$epoch"
done
touch "$EVAL_ROOT/GPU0_AUX_COMPLETE"
echo "[gpu0-eval] $(date '+%F %T') auxiliary evaluations complete"
