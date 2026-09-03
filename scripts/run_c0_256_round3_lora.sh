#!/usr/bin/env bash
set -euo pipefail
cd /Data_8TB/lht/PseudoVideo-SAM3-X3-B7
PY=/home/violet/anaconda3/envs/sam3/bin/python
CFG=configs/c0_256_round3_tracker_lora.yaml
ROOT=work/rerun_c0_256_round3_tracker_lora_t3_t4
ACTION=${1:-prepare}
DEVICE=${2:-0}

prepare() {
  mkdir -p "$ROOT"
  "$PY" scripts/audit_round3_supervision.py --config "$CFG"
  "$PY" -m pytest -q tests/test_round3_lora.py tests/test_round3_no_split_leakage.py tests/test_round3_sequence_manifest.py tests/test_round3_checkpoint_reload.py
  CUDA_VISIBLE_DEVICES="$DEVICE" "$PY" scripts/train_round3_tracker.py --config "$CFG" --group T3 --device "$DEVICE" --audit-only
  CUDA_VISIBLE_DEVICES="$DEVICE" "$PY" scripts/train_round3_tracker.py --config "$CFG" --group T4 --device "$DEVICE" --audit-only
}

smoke_group() {
  group=$1
  run_name=$2
  CUDA_VISIBLE_DEVICES="$DEVICE" "$PY" scripts/train_round3_tracker.py --config "$CFG" --group "$group" --device "$DEVICE" --smoke --max-steps 10 > "$ROOT/$run_name/smoke.log" 2>&1
}

train_group() {
  group=$1
  run_name=$2
  CUDA_VISIBLE_DEVICES="$DEVICE" "$PY" scripts/train_round3_tracker.py --config "$CFG" --group "$group" --device "$DEVICE" > "$ROOT/$run_name/train.log" 2>&1
}

validate_group() {
  group=$1
  run_name=$2
  for checkpoint in "$ROOT/$run_name/checkpoints"/step_*.pt; do
    tag=$(basename "$checkpoint" .pt)
    CUDA_VISIBLE_DEVICES="$DEVICE" "$PY" scripts/eval_round3_tracker.py --config "$CFG" --group "$group" --split validation --tracker-checkpoint "$checkpoint" --device "$DEVICE" --tag "$tag" --resume
    CUDA_VISIBLE_DEVICES="$DEVICE" "$PY" scripts/eval_round3_path_sensitivity.py --config "$CFG" --group "$group" --tracker-checkpoint "$checkpoint" --device "$DEVICE" --tag "$tag" --resume
  done
  "$PY" scripts/summarize_round3_lora.py --config "$CFG" --group "$group"
}

test_group() {
  group=$1
  run_name=$2
  best=$($PY -c "import json; print(json.load(open('$ROOT/$run_name/best_checkpoint.json'))['checkpoint'])")
  CUDA_VISIBLE_DEVICES="$DEVICE" "$PY" scripts/eval_round3_tracker.py --config "$CFG" --group "$group" --split test --tracker-checkpoint "$best" --device "$DEVICE" --tag best --resume
}

case "$ACTION" in
  prepare) prepare ;;
  smoke-t3) smoke_group T3 T3_memory_attention_lora ;;
  smoke-t4) smoke_group T4 T4_memory_attention_decoder_lora ;;
  train-t3) train_group T3 T3_memory_attention_lora ;;
  train-t4) train_group T4 T4_memory_attention_decoder_lora ;;
  validate-t3) validate_group T3 T3_memory_attention_lora > "$ROOT/T3_memory_attention_lora/validation_queue.log" 2>&1 ;;
  validate-t4) validate_group T4 T4_memory_attention_decoder_lora > "$ROOT/T4_memory_attention_decoder_lora/validation_queue.log" 2>&1 ;;
  test-t3) test_group T3 T3_memory_attention_lora > "$ROOT/T3_memory_attention_lora/test_best.log" 2>&1 ;;
  test-t4) test_group T4 T4_memory_attention_decoder_lora > "$ROOT/T4_memory_attention_decoder_lora/test_best.log" 2>&1 ;;
  closeout) "$PY" scripts/summarize_round3_lora.py --config "$CFG" --closeout; "$PY" scripts/finalize_round3_lora_report.py; touch "$ROOT/ROUND3_LORA_COMPLETE" ;;
  *) echo "usage: $0 {prepare|smoke-t3|smoke-t4|train-t3|train-t4|validate-t3|validate-t4|test-t3|test-t4|closeout} [device]" >&2; exit 2 ;;
esac
