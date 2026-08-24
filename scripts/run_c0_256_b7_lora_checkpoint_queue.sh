#!/usr/bin/env bash
# Evaluate LoRA epochs on validation using the frozen C0-256 X3-best+B7 rule.
# Only the validation-best epoch is evaluated on test after the queue finishes.
set -euo pipefail

cd /Data_8TB/lht/PseudoVideo-SAM3-X3-B7

PY=/home/violet/anaconda3/envs/sam3/bin/python
BASE=/Data_8TB/lht/models/modelscope/models/facebook--sam3/snapshots/master/sam3.pt
RUN=work/rerun_c0_256_base/medsam3_lora_b7_e50
LORA_DIR="$RUN/lora_weights"
EVAL_ROOT="$RUN/checkpoint_eval"
ROUTE_SOURCE=work/rerun_c0_256_base/stage1_feature_knn_b7_base
FILTERED_ROUTES="$EVAL_ROOT/routes_b3_b6"
VAL_PRED=work/rerun_c0_256_base/predictions/X3_best/student_predictions_validation.jsonl
TEST_PRED=work/rerun_c0_256_base/predictions/X3_best/student_predictions_test.jsonl
DEVICE=${DEVICE:-1}
export CUDA_VISIBLE_DEVICES="$DEVICE"

MODES=(
  anchor_conditioned_target_pooling
  anchor_conditioned_patch_correspondence
)

# Every currently available epoch (e1-e12), then spaced checkpoints through e50.
NODES=(1 2 3 4 5 6 7 8 9 10 11 12 15 20 25 30 35 40 45 50)

mkdir -p "$EVAL_ROOT/merged" "$FILTERED_ROUTES"

prepare_routes() {
  local split=$1 mode src dst
  for mode in "${MODES[@]}"; do
    src="$ROUTE_SOURCE/$mode/${split}_pool0_stage1/routes.jsonl"
    dst="$FILTERED_ROUTES/$mode/${split}_pool0_stage1/routes.jsonl"
    if [[ ! -f "$dst" ]]; then
      "$PY" scripts/filter_route_pool.py \
        --input "$src" --output "$dst" --min-bridge 3 --max-bridge 6
    fi
  done
}

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
  lora="$LORA_DIR/epoch_${epoch}_lora_weights.pt"
  merged="$EVAL_ROOT/merged/${tag}_merged_video.pt"
  qroot="$EVAL_ROOT/$tag/quality_root"

  while [[ ! -f "$lora" ]]; do
    echo "[queue] $(date '+%F %T') waiting for $lora"
    sleep 300
  done
  if [[ -f "$EVAL_ROOT/$tag/VALIDATION_COMPLETE" ]]; then
    echo "[queue] $tag validation already complete"
    summarize_completed
    return
  fi

  echo "[queue] $(date '+%F %T') merge $tag"
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
    echo "[queue] $(date '+%F %T') $tag validation $mode"
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
  echo "[queue] $(date '+%F %T') $tag validation complete"
}

evaluate_test_best() {
  local epoch tag merged qroot mode
  epoch=$(cat "$EVAL_ROOT/best_validation_epoch.txt")
  printf -v tag 'e%02d' "$epoch"
  merged="$EVAL_ROOT/merged/${tag}_merged_video.pt"
  qroot="$EVAL_ROOT/$tag/quality_root"
  echo "[queue] validation-best epoch is $epoch"

  for mode in "${MODES[@]}"; do
    mkdir -p "$qroot/$mode"
    ln -sfn "$(realpath "$FILTERED_ROUTES/$mode/test_pool0_stage1")" \
      "$qroot/$mode/test_pool0_stage1"
    "$PY" scripts/eval_route_propagation_quality.py \
      --checkpoint "$merged" \
      --mode "$mode" \
      --root "$qroot" \
      --split test \
      --canvas 256 \
      --resume
  done
  "$PY" scripts/phase1_b7_select.py \
    --quality-root "$qroot" \
    --student-predictions "$TEST_PRED" \
    --output-dir "$EVAL_ROOT/test_validation_best_b7"
  touch "$EVAL_ROOT/TEST_VALIDATION_BEST_COMPLETE"
}

prepare_routes validation
prepare_routes test
for epoch in "${NODES[@]}"; do
  evaluate_validation_epoch "$epoch"
done
summarize_completed
evaluate_test_best
touch "$EVAL_ROOT/QUEUE_COMPLETE"
echo "[queue] $(date '+%F %T') all checkpoint evaluations complete"
