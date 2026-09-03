#!/usr/bin/env bash
set -euo pipefail
cd /Data_8TB/lht/PseudoVideo-SAM3-X3-B7
ROOT=work/rerun_c0_256_round3_tracker_stage4
T1=$ROOT/T1_memory/checkpoints/step_020000.pt
T2=$ROOT/T2_full_tracker/checkpoints/step_020000.pt
while [[ ! -f "$T1" || ! -f "$T2" ]]; do
  t1_running=$(pgrep -fc 'train_round3_tracker.py.*--group T1' || true)
  t2_running=$(pgrep -fc 'train_round3_tracker.py.*--group T2' || true)
  if [[ "$t1_running" -eq 0 || "$t2_running" -eq 0 ]]; then
    echo "training exited before both 20k checkpoints existed" >&2
    exit 1
  fi
  echo "$(date -Is) waiting for formal training; T1=$([[ -f "$T1" ]] && echo done || echo running) T2=$([[ -f "$T2" ]] && echo done || echo running)"
  sleep 60
done
bash scripts/run_c0_256_round3_tracker.sh validate
bash scripts/run_c0_256_round3_tracker.sh test
/home/violet/anaconda3/envs/sam3/bin/python scripts/finalize_round3_report.py
touch "$ROOT/ROUND3_STAGE4_COMPLETE"
echo "$(date -Is) ROUND3_STAGE4_COMPLETE"
