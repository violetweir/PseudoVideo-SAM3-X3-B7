#!/usr/bin/env bash
set -euo pipefail

cd /Data_8TB/lht/PseudoVideo-SAM3-X3-B7
LOG=work/clinicdb_external_kvasir8/queued_run.log
mkdir -p "$(dirname "$LOG")"

while pgrep -f '[r]un_route_selector_vitb256.sh' >/dev/null; do
  echo "[queue] waiting for run_route_selector_vitb256.sh $(date '+%F %T')" >> "$LOG"
  sleep 60
done

echo "[queue] starting ClinicDB external pipeline $(date '+%F %T')" >> "$LOG"
bash scripts/run_clinicdb_external_pairwise_vitb256.sh >> "$LOG" 2>&1
echo "[queue] finished ClinicDB external pipeline $(date '+%F %T')" >> "$LOG"
