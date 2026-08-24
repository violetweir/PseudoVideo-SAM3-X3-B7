#!/usr/bin/env bash
set -u

cd /Data_8TB/lht/PseudoVideo-SAM3-X3-B7
RUN=work/rerun_c0_256_base/medsam3_lora_b7_e50
EVAL_ROOT="$RUN/checkpoint_eval"
LOG="$RUN/checkpoint_eval_queue.log"
SUPERVISOR_LOG="$RUN/checkpoint_eval_supervisor.log"
DEVICE=${DEVICE:-1}

mkdir -p "$EVAL_ROOT"
while [[ ! -f "$EVAL_ROOT/QUEUE_COMPLETE" ]]; do
  echo "[eval-supervisor] $(date '+%F %T') launch queue on GPU $DEVICE" | tee -a "$SUPERVISOR_LOG"
  env DEVICE="$DEVICE" bash scripts/run_c0_256_b7_lora_checkpoint_queue.sh >> "$LOG" 2>&1
  status=$?
  if [[ $status -eq 0 ]]; then
    break
  fi
  echo "[eval-supervisor] $(date '+%F %T') queue exited status=$status; resume in 30s" | tee -a "$SUPERVISOR_LOG"
  sleep 30
done
echo "[eval-supervisor] $(date '+%F %T') complete" | tee -a "$SUPERVISOR_LOG"
