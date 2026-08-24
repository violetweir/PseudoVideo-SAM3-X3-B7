#!/usr/bin/env bash
set -Eeuo pipefail

cd /Data_8TB/lht/PseudoVideo-SAM3-X3-B7

PY=/home/violet/anaconda3/envs/mkunet_mamba/bin/python
PHASE=work/rerun_c0_256_sam3knn_s256_base
ROOT=$PHASE/stage1_feature_knn_b0_b6
DATA=work/kvasir_1pct_anchors/baseline_data
LABELS=work/kvasir_1pct_anchors/protocol/frozen_labeled_images.txt
log=$PHASE/audit_x3.log
failed_marker=$PHASE/AUDIT_X3_FAILED

rm -f "$failed_marker"
trap 'touch "$failed_marker"' ERR

while [[ ! -f $PHASE/S2_EXPORT_COMPLETE || ! -f $PHASE/S3_EXPORT_COMPLETE ]]; do
  if [[ -f $PHASE/S2_EXPORT_FAILED || -f $PHASE/S3_EXPORT_FAILED ]]; then
    echo "[audit-x3] student export failed" | tee -a "$log"
    exit 1
  fi
  sleep 30
done

echo "[audit-x3] $(date '+%F %T') build audit" | tee -a "$log"
"$PY" scripts/phase1_audit_tiers.py \
  --train-metadata "$DATA/train/metadata.jsonl" \
  --labeled-list "$LABELS" \
  --quality-root "$ROOT" \
  --original-manifest "$PHASE/pseudo_manifest_original_b0_b6.jsonl" \
  --predictions "$PHASE/predictions/S2_valbest/student_predictions_train.jsonl" --predictions-name S2_valbest \
  --predictions "$PHASE/predictions/S2_final/student_predictions_train.jsonl" --predictions-name S2_final \
  --predictions "$PHASE/predictions/S3_valbest/student_predictions_train.jsonl" --predictions-name S3_valbest \
  --predictions "$PHASE/predictions/S3_final/student_predictions_train.jsonl" --predictions-name S3_final \
  --output-dir "$PHASE/audit" \
  2>&1 | tee -a "$log"
touch "$PHASE/AUDIT_COMPLETE"

echo "[audit-x3] $(date '+%F %T') build X3 manifest" | tee -a "$log"
"$PY" scripts/phase1_build_x3_manifest.py \
  --original "$PHASE/pseudo_manifest_original_b0_b6.jsonl" \
  --tier-a "$PHASE/audit/tier_A.jsonl" \
  --tier-b "$PHASE/audit/tier_B.jsonl" \
  --output "$PHASE/pseudo_manifest_x3.jsonl" \
  2>&1 | tee -a "$log"
touch "$PHASE/X3_MANIFEST_COMPLETE"

echo "[audit-x3] $(date '+%F %T') start X3 gpu=0" | tee -a "$log"
CUDA_VISIBLE_DEVICES=0 "$PY" scripts/run_s27_student.py \
  --data-path "$DATA" --labeled-list "$LABELS" \
  --pseudo-manifest "$PHASE/pseudo_manifest_x3.jsonl" \
  --output-dir "$PHASE/students/X3" --experiment X3 --seed 2026 \
  --batch-size 12 --gt-bs 3 --original-bs 3 --new-bs 6 \
  --max-iterations 40000 --val-interval 200 --num-workers 4 \
  2>&1 | tee -a "$PHASE/x3_train.log"
touch "$PHASE/X3_TRAIN_COMPLETE"
echo "[audit-x3] $(date '+%F %T') X3 complete" | tee -a "$log"
