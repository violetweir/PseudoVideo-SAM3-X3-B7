#!/usr/bin/env bash
# Keep the 50-epoch C0-256-base X3+B7 full-module LoRA run alive.
set -u

cd /Data_8TB/lht/PseudoVideo-SAM3-X3-B7

OUT=work/rerun_c0_256_base/medsam3_lora_b7_e50/lora_weights
CONFIG=configs/c0_256_base_b7_medsam3_lora_e50.yaml
LOG=work/rerun_c0_256_base/medsam3_lora_b7_e50/train.log
SUPERVISOR_LOG=work/rerun_c0_256_base/medsam3_lora_b7_e50/supervisor.log
FINAL="$OUT/epoch_50_lora_weights.pt"
PY=/home/violet/anaconda3/envs/sam3/bin/python
TRAINER=scripts/train_sam3_lora_kvasir_e50.py
DEVICE=${DEVICE:-0}

mkdir -p "$OUT"

latest_epoch() {
  local latest
  latest=$(find "$OUT" -maxdepth 1 -type f -name 'epoch_*_lora_weights.pt' -printf '%f\n' 2>/dev/null \
    | sed -E 's/epoch_([0-9]+)_lora_weights.pt/\1/' \
    | sort -n \
    | tail -1)
  printf '%s' "$latest"
}

launch_from_latest() {
  local epoch resume
  epoch=$(latest_epoch)
  if [[ -z "$epoch" ]]; then
    echo "[supervisor] $(date '+%F %T') launch from epoch 0 on GPU $DEVICE" | tee -a "$SUPERVISOR_LOG"
    "$PY" "$TRAINER" --config "$CONFIG" --device "$DEVICE" >> "$LOG" 2>&1 &
  else
    resume="$OUT/epoch_${epoch}_lora_weights.pt"
    echo "[supervisor] $(date '+%F %T') resume after epoch $epoch from $resume on GPU $DEVICE" | tee -a "$SUPERVISOR_LOG"
    "$PY" "$TRAINER" --config "$CONFIG" --device "$DEVICE" \
      --resume-lora "$resume" --start-epoch "$epoch" >> "$LOG" 2>&1 &
  fi
  TRAIN_PID=$!
  echo "[supervisor] child pid=$TRAIN_PID" | tee -a "$SUPERVISOR_LOG"
}

if [[ -f "$FINAL" ]]; then
  echo "[supervisor] final checkpoint already exists: $FINAL" | tee -a "$SUPERVISOR_LOG"
  exit 0
fi

launch_from_latest
while [[ ! -f "$FINAL" ]]; do
  if kill -0 "$TRAIN_PID" 2>/dev/null; then
    sleep 60
    continue
  fi
  echo "[supervisor] $(date '+%F %T') child ended; relaunching" | tee -a "$SUPERVISOR_LOG"
  sleep 10
  launch_from_latest
done

echo "[supervisor] $(date '+%F %T') epoch 50 saved; complete" | tee -a "$SUPERVISOR_LOG"
