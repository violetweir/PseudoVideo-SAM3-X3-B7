#!/usr/bin/env bash
set -Eeuo pipefail

cd /Data_8TB/lht/PseudoVideo-SAM3-X3-B7

PY=/home/violet/anaconda3/envs/mkunet_mamba/bin/python
PHASE=work/rerun_c0_256_sam3knn_s256_base
DATA=work/kvasir_1pct_anchors/baseline_data
LABELS=work/kvasir_1pct_anchors/protocol/frozen_labeled_images.txt

student=${1:?usage: $0 S2|S3 GPU_ID}
gpu=${2:?usage: $0 S2|S3 GPU_ID}

case "$student" in
  S2)
    manifest=$PHASE/pseudo_manifest_original_b0_b6.jsonl
    ;;
  S3)
    manifest=$PHASE/S3_consensus_b0_b6/pseudo_consensus.jsonl
    ;;
  *)
    echo "unknown student: $student" >&2
    exit 2
    ;;
esac

out=$PHASE/students/$student
log=$PHASE/${student,,}_train.log
done_marker=$PHASE/${student}_TRAIN_COMPLETE
failed_marker=$PHASE/${student}_TRAIN_FAILED

mkdir -p "$out"
rm -f "$failed_marker"
trap 'touch "$failed_marker"' ERR

echo "[$student] $(date '+%F %T') start gpu=$gpu manifest=$manifest" | tee -a "$log"
CUDA_VISIBLE_DEVICES="$gpu" "$PY" scripts/run_t24_student.py \
  --data-path "$DATA" --labeled-list "$LABELS" \
  --pseudo-manifest "$manifest" \
  --output-dir "$out" --experiment "$student" --seed 2026 \
  --max-iterations 40000 --val-interval 200 --num-workers 4 \
  2>&1 | tee -a "$log"

touch "$done_marker"
echo "[$student] $(date '+%F %T') complete" | tee -a "$log"
