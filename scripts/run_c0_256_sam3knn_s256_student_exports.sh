#!/usr/bin/env bash
set -Eeuo pipefail

cd /Data_8TB/lht/PseudoVideo-SAM3-X3-B7

PY=/home/violet/anaconda3/envs/mkunet_mamba/bin/python
PHASE=work/rerun_c0_256_sam3knn_s256_base
student=${1:?usage: $0 S2|S3 GPU_ID}
gpu=${2:?usage: $0 S2|S3 GPU_ID}
log=$PHASE/${student,,}_export.log
done_marker=$PHASE/${student}_EXPORT_COMPLETE
failed_marker=$PHASE/${student}_EXPORT_FAILED

case "$student" in
  S2|S3) ;;
  *) echo "unknown student: $student" >&2; exit 2 ;;
esac

rm -f "$failed_marker"
trap 'touch "$failed_marker"' ERR

for entry in \
  "student_best.pth ${student}_valbest" \
  "student_final.pth ${student}_final"; do
  read -r checkpoint output_name <<<"$entry"
  echo "[$student] $(date '+%F %T') export $checkpoint" | tee -a "$log"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" scripts/export_t25_student_predictions.py \
    --run-dir "$PHASE/students/$student" \
    --checkpoint "$PHASE/students/$student/$checkpoint" \
    --output-root "$PHASE/predictions/$output_name" \
    --splits train validation \
    2>&1 | tee -a "$log"
done

touch "$done_marker"
echo "[$student] $(date '+%F %T') export complete" | tee -a "$log"
