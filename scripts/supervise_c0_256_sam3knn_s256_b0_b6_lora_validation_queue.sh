#!/usr/bin/env bash
set -u

cd /Data_8TB/lht/PseudoVideo-SAM3-X3-B7

RUN=work/rerun_c0_256_sam3knn_s256_base/medsam3_lora_b0_b6_e50
EVAL_ROOT=$RUN/checkpoint_eval
LOG=$RUN/checkpoint_eval_queue.log
SUPERVISOR_LOG=$RUN/checkpoint_eval_supervisor.log
DEVICE=${DEVICE:-1}

mkdir -p "$EVAL_ROOT"
while [[ ! -f $EVAL_ROOT/VALIDATION_QUEUE_COMPLETE ]]; do
  echo "[queue-supervisor] $(date '+%F %T') launch GPU $DEVICE" | tee -a "$SUPERVISOR_LOG"
  env DEVICE="$DEVICE" bash scripts/run_c0_256_sam3knn_s256_b0_b6_lora_validation_queue.sh \
    >> "$LOG" 2>&1
  status=$?
  if [[ $status -eq 0 ]]; then
    break
  fi
  echo "[queue-supervisor] $(date '+%F %T') exit=$status; retry in 30s" | tee -a "$SUPERVISOR_LOG"
  sleep 30
done

echo "[queue-supervisor] $(date '+%F %T') complete" | tee -a "$SUPERVISOR_LOG"
