#!/usr/bin/env bash
set -euo pipefail
cd /Data_8TB/lht/PseudoVideo-SAM3-X3-B7
PY=/home/violet/anaconda3/envs/sam3/bin/python
CFG=configs/c0_256_round3_tracker_stage4.yaml
ROOT=work/rerun_c0_256_round3_tracker_stage4
ACTION=${1:-prepare}

prepare() {
  "$PY" scripts/build_round3_tracker_sequences.py --config "$CFG"
  "$PY" -m pytest -q tests/test_round3_no_split_leakage.py tests/test_round3_sequence_manifest.py tests/test_round3_freeze_boundary.py tests/test_round3_checkpoint_reload.py
  CUDA_VISIBLE_DEVICES=0 "$PY" scripts/train_round3_tracker.py --config "$CFG" --group T2 --device 0 --cache-only
  CUDA_VISIBLE_DEVICES=0 "$PY" scripts/train_round3_tracker.py --config "$CFG" --group T1 --device 0 --audit-only
  CUDA_VISIBLE_DEVICES=0 "$PY" scripts/train_round3_tracker.py --config "$CFG" --group T2 --device 0 --audit-only
}

t0() {
  CUDA_VISIBLE_DEVICES=0 "$PY" scripts/eval_round3_tracker.py --config "$CFG" --group T0 --split validation --device 0 --resume
  CUDA_VISIBLE_DEVICES=0 "$PY" scripts/eval_round3_tracker.py --config "$CFG" --group T0 --split test --device 0 --resume
}

smoke() {
  CUDA_VISIBLE_DEVICES=0 "$PY" scripts/train_round3_tracker.py --config "$CFG" --group T1 --device 0 --smoke --max-steps 10 > "$ROOT/T1_memory/smoke.log" 2>&1 &
  p1=$!
  CUDA_VISIBLE_DEVICES=1 "$PY" scripts/train_round3_tracker.py --config "$CFG" --group T2 --device 1 --smoke --max-steps 10 > "$ROOT/T2_full_tracker/smoke.log" 2>&1 &
  p2=$!
  wait "$p1"
  wait "$p2"
}

train() {
  CUDA_VISIBLE_DEVICES=0 "$PY" scripts/train_round3_tracker.py --config "$CFG" --group T1 --device 0 > "$ROOT/T1_memory/train.log" 2>&1 &
  p1=$!
  CUDA_VISIBLE_DEVICES=1 "$PY" scripts/train_round3_tracker.py --config "$CFG" --group T2 --device 1 > "$ROOT/T2_full_tracker/train.log" 2>&1 &
  p2=$!
  wait "$p1"
  wait "$p2"
}

validate_group() {
  group=$1
  device=$2
  run_name=$3
  for checkpoint in "$ROOT/$run_name/checkpoints"/step_*.pt; do
    tag=$(basename "$checkpoint" .pt)
    CUDA_VISIBLE_DEVICES="$device" "$PY" scripts/eval_round3_tracker.py --config "$CFG" --group "$group" --split validation --tracker-checkpoint "$checkpoint" --device "$device" --tag "$tag" --resume
  done
  "$PY" scripts/summarize_round3_tracker.py --config "$CFG" --group "$group"
}

validate() {
  validate_group T1 0 T1_memory > "$ROOT/T1_memory/validation_queue.log" 2>&1 &
  p1=$!
  validate_group T2 1 T2_full_tracker > "$ROOT/T2_full_tracker/validation_queue.log" 2>&1 &
  p2=$!
  wait "$p1"
  wait "$p2"
}

test_best() {
  t1=$("$PY" -c "import json; print(json.load(open('$ROOT/T1_memory/best_checkpoint.json'))['checkpoint'])")
  t2=$("$PY" -c "import json; print(json.load(open('$ROOT/T2_full_tracker/best_checkpoint.json'))['checkpoint'])")
  CUDA_VISIBLE_DEVICES=0 "$PY" scripts/eval_round3_tracker.py --config "$CFG" --group T1 --split test --tracker-checkpoint "$t1" --device 0 --tag best --resume > "$ROOT/T1_memory/test_best.log" 2>&1 &
  p1=$!
  CUDA_VISIBLE_DEVICES=1 "$PY" scripts/eval_round3_tracker.py --config "$CFG" --group T2 --split test --tracker-checkpoint "$t2" --device 1 --tag best --resume > "$ROOT/T2_full_tracker/test_best.log" 2>&1 &
  p2=$!
  wait "$p1"
  wait "$p2"
  "$PY" scripts/summarize_round3_tracker.py --config "$CFG" --closeout
}

case "$ACTION" in
  prepare) prepare ;;
  t0) t0 ;;
  smoke) smoke ;;
  train) train ;;
  validate) validate ;;
  test) test_best ;;
  all) prepare; t0; smoke; train; validate; test_best ;;
  *) echo "usage: $0 {prepare|t0|smoke|train|validate|test|all}" >&2; exit 2 ;;
esac
