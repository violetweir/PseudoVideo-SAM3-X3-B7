#!/usr/bin/env bash
# Keep Round-2A progressing from e33 propagation through X4 comparison.
set -u

cd /Data_8TB/lht/PseudoVideo-SAM3-X3-B7

PHASE=work/rerun_c0_256_round2a_fixed_knn_e33
LOG=$PHASE/pipeline_supervisor.log
MKPY=/home/violet/anaconda3/envs/mkunet_mamba/bin/python
mkdir -p "$PHASE"

while [[ ! -f "$PHASE/TRAIN_PROP_COMPLETE" ]]; do
  if [[ -f "$PHASE/TRAIN_PROP_FAILED" ]]; then
    echo "[supervisor] $(date '+%F %T') propagation failed; resume on GPU0" | tee -a "$LOG"
    rm -f "$PHASE/TRAIN_PROP_FAILED"
    DEVICE=0 bash scripts/run_c0_256_round2a_e33_train_propagation.sh \
      >> "$PHASE/nohup_train_propagation.log" 2>&1
    continue
  fi
  if ! pgrep -f '[r]un_c0_256_round2a_e33_train_propagation.sh' >/dev/null; then
    echo "[supervisor] $(date '+%F %T') propagation process absent; resume" | tee -a "$LOG"
    DEVICE=0 bash scripts/run_c0_256_round2a_e33_train_propagation.sh \
      >> "$PHASE/nohup_train_propagation.log" 2>&1
    continue
  fi
  sleep 60
done

while [[ ! -f "$PHASE/B7_CALIBRATION_COMPLETE" ]]; do
  echo "[supervisor] $(date '+%F %T') waiting for B7 calibration" | tee -a "$LOG"
  sleep 30
done

while [[ ! -f "$PHASE/X4_TRAIN_COMPLETE" ]]; do
  echo "[supervisor] $(date '+%F %T') build pool / train X4" | tee -a "$LOG"
  if DEVICE=0 bash scripts/run_c0_256_round2a_build_pool_and_x4.sh \
      >> "$PHASE/x4_pipeline.log" 2>&1; then
    break
  fi
  echo "[supervisor] $(date '+%F %T') X4 pipeline ended; resume in 30s" | tee -a "$LOG"
  sleep 30
done

if [[ ! -f "$PHASE/X3_X4_TEST_COMPLETE" ]]; then
  echo "[supervisor] $(date '+%F %T') evaluate X3 vs X4" | tee -a "$LOG"
  CUDA_VISIBLE_DEVICES=0 "$MKPY" scripts/eval_c0_256_round2a_x3_x4.py \
    --x3-run work/rerun_c0_256_sam3knn_s256_base/students/X3 \
    --x4-run "$PHASE/students/X4" \
    --output "$PHASE/x3_vs_x4.json" \
    >> "$PHASE/x3_vs_x4.log" 2>&1
  if [[ $? -eq 0 ]]; then
    touch "$PHASE/X3_X4_TEST_COMPLETE"
  fi
fi

if [[ -f "$PHASE/X3_X4_TEST_COMPLETE" ]]; then
  touch "$PHASE/ROUND2A_COMPLETE"
  echo "[supervisor] $(date '+%F %T') Round-2A complete" | tee -a "$LOG"
fi
