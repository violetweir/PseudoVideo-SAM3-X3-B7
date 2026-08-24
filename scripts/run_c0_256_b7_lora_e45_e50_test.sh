#!/usr/bin/env bash
# Test late LoRA checkpoints with the same frozen C0-256 X3-best+B7 protocol.
set -euo pipefail

cd /Data_8TB/lht/PseudoVideo-SAM3-X3-B7

PY=/home/violet/anaconda3/envs/sam3/bin/python
RUN=work/rerun_c0_256_base/medsam3_lora_b7_e50
EVAL_ROOT="$RUN/checkpoint_eval"
FILTERED_ROUTES="$EVAL_ROOT/routes_b3_b6"
TEST_PRED=work/rerun_c0_256_base/predictions/X3_best/student_predictions_test.jsonl
export CUDA_VISIBLE_DEVICES=1

MODES=(
  anchor_conditioned_target_pooling
  anchor_conditioned_patch_correspondence
)
NODES=(45 50)

for epoch in "${NODES[@]}"; do
  printf -v tag 'e%02d' "$epoch"
  merged="$EVAL_ROOT/merged/${tag}_merged_video.pt"
  qroot="$EVAL_ROOT/$tag/quality_root"
  output_dir="$EVAL_ROOT/test_${tag}_b7"

  if [[ -f "$output_dir/TEST_COMPLETE" ]]; then
    echo "[late-test] $(date '+%F %T') $tag already complete"
    continue
  fi
  if [[ ! -f "$merged" ]]; then
    echo "[late-test] missing merged checkpoint: $merged" >&2
    exit 1
  fi

  for mode in "${MODES[@]}"; do
    mkdir -p "$qroot/$mode"
    ln -sfn "$(realpath "$FILTERED_ROUTES/$mode/test_pool0_stage1")" \
      "$qroot/$mode/test_pool0_stage1"
    echo "[late-test] $(date '+%F %T') $tag test $mode"
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
    --output-dir "$output_dir"
  touch "$output_dir/TEST_COMPLETE"
  echo "[late-test] $(date '+%F %T') $tag test complete"
done

touch "$EVAL_ROOT/E45_E50_TEST_COMPLETE"
echo "[late-test] $(date '+%F %T') all late checkpoint tests complete"
