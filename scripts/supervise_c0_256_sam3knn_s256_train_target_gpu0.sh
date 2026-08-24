#!/usr/bin/env bash
set -u

cd /Data_8TB/lht/PseudoVideo-SAM3-X3-B7
PHASE=work/rerun_c0_256_sam3knn_s256_base
LOG="$PHASE/train_target_gpu0_supervisor.log"
DONE="$PHASE/TARGET_TRAIN_PROP_COMPLETE"

while [[ ! -f "$DONE" ]]; do
  echo "[target-gpu0-supervisor] $(date '+%F %T') launch" | tee -a "$LOG"
  bash scripts/run_c0_256_sam3knn_s256_train_target_gpu0.sh
  status=$?
  if [[ -f "$DONE" ]]; then
    break
  fi
  echo "[target-gpu0-supervisor] $(date '+%F %T') status=$status; resume in 60s" | tee -a "$LOG"
  sleep 60
done

echo "[target-gpu0-supervisor] $(date '+%F %T') complete" | tee -a "$LOG"
