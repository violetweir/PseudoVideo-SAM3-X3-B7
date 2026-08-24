#!/usr/bin/env bash
set -u

cd /Data_8TB/lht/PseudoVideo-SAM3-X3-B7
RUN=work/rerun_c0_256_base/medsam3_lora_b7_e50
LOG="$RUN/checkpoint_eval_gpu0_aux.log"
DONE="$RUN/checkpoint_eval/GPU0_AUX_COMPLETE"

while [[ ! -f "$DONE" ]]; do
  echo "[gpu0-supervisor] $(date '+%F %T') launch auxiliary queue" | tee -a "$LOG"
  bash scripts/run_c0_256_b7_lora_gpu0_aux.sh >> "$LOG" 2>&1
  status=$?
  if [[ -f "$DONE" ]]; then
    break
  fi
  echo "[gpu0-supervisor] $(date '+%F %T') exited status=$status; resume in 30s" | tee -a "$LOG"
  sleep 30
done

echo "[gpu0-supervisor] $(date '+%F %T') complete" | tee -a "$LOG"
