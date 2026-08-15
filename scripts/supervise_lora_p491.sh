#!/usr/bin/env bash
# Supervisor for A-2 (LoRA 1% + 491): keeps the training process alive and
# restarts from the latest saved epoch if it is killed (session reaping, etc.).
#
# Resume mapping: epoch_N_lora_weights.pt is saved at the END of 0-based epoch
# N-1, so the next 0-based epoch to run is N.
set -u

cd /Data_8TB/lht/PseudoVideo-SAM3-X3-B7
OUT=work/kvasir_1pct_anchors/lora_experiment/lora_1pct_plus_pseudo491
CONFIG=configs/kvasir_1pct_plus_pseudo491_lora.yaml
LOG=work/kvasir_1pct_anchors/lora_experiment/train_lora_p491_supervise.log
FINAL="$OUT/epoch_20_lora_weights.pt"
PY=/home/violet/anaconda3/envs/sam3/bin/python

latest_epoch() {
  ls "$OUT"/epoch_*_lora_weights.pt 2>/dev/null \
    | sed -E 's/.*epoch_([0-9]+)_lora.*/\1/' \
    | sort -n | tail -1
}

launch() {
  local start_epoch="$1"
  local resume="$2"
  echo "[supervisor] $(date '+%F %T') launching start_epoch=$start_epoch resume=$resume"
  if [[ -n "$resume" ]]; then
    CUDA_VISIBLE_DEVICES=0 "$PY" scripts/train_sam3_lora_kvasir.py \
      --config "$CONFIG" --resume-lora "$resume" --start-epoch "$start_epoch" >> "$LOG" 2>&1 &
  else
    CUDA_VISIBLE_DEVICES=0 "$PY" scripts/train_sam3_lora_kvasir.py \
      --config "$CONFIG" >> "$LOG" 2>&1 &
  fi
  TRAIN_PID=$!
  echo "[supervisor] pid=$TRAIN_PID"
}

launch_from_latest() {
  local e
  e=$(latest_epoch)
  if [[ -z "$e" ]]; then
    launch 0 ""
  else
    launch "$e" "$OUT/epoch_${e}_lora_weights.pt"
  fi
}

echo "[supervisor] $(date '+%F %T') starting, final=$FINAL"
launch_from_latest

while [[ ! -f "$FINAL" ]]; do
  if kill -0 "$TRAIN_PID" 2>/dev/null; then
    sleep 60
    continue
  fi
  echo "[supervisor] $(date '+%F %T') training process ended (pid=$TRAIN_PID); restarting"
  launch_from_latest
  sleep 30
done

echo "[supervisor] $(date '+%F %T') final epoch saved; done"
