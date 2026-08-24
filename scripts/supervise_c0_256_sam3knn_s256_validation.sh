#!/usr/bin/env bash
set -u

cd /Data_8TB/lht/PseudoVideo-SAM3-X3-B7
PHASE=work/rerun_c0_256_sam3knn_s256_base
LOG="$PHASE/validation_supervisor.log"
DONE="$PHASE/VALIDATION_COMPLETE"

while [[ ! -f "$DONE" ]]; do
  echo "[validation-supervisor] $(date '+%F %T') launch" | tee -a "$LOG"
  bash scripts/run_c0_256_sam3knn_s256_validation.sh
  status=$?
  if [[ -f "$DONE" ]]; then
    break
  fi
  echo "[validation-supervisor] $(date '+%F %T') status=$status; resume in 30s" | tee -a "$LOG"
  sleep 30
done

echo "[validation-supervisor] $(date '+%F %T') complete" | tee -a "$LOG"
