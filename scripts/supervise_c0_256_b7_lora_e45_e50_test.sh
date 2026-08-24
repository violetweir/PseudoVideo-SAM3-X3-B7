#!/usr/bin/env bash
set -u

cd /Data_8TB/lht/PseudoVideo-SAM3-X3-B7
RUN=work/rerun_c0_256_base/medsam3_lora_b7_e50
LOG="$RUN/checkpoint_eval_e45_e50_test.log"
DONE="$RUN/checkpoint_eval/E45_E50_TEST_COMPLETE"

while [[ ! -f "$DONE" ]]; do
  echo "[late-test-supervisor] $(date '+%F %T') launch on GPU1" | tee -a "$LOG"
  bash scripts/run_c0_256_b7_lora_e45_e50_test.sh >> "$LOG" 2>&1
  status=$?
  if [[ -f "$DONE" ]]; then
    break
  fi
  echo "[late-test-supervisor] $(date '+%F %T') exited status=$status; resume in 30s" | tee -a "$LOG"
  sleep 30
done

echo "[late-test-supervisor] $(date '+%F %T') complete" | tee -a "$LOG"
