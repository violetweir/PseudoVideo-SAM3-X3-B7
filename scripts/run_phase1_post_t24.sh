#!/usr/bin/env bash
# Phase-1 post-T24: exports -> committee audit -> X3 manifest -> X3 -> B7.
# Expects T24 S2 and S3 training complete under phase1/students/{S2,S3}.
set -euo pipefail

cd /Data_8TB/lht/PseudoVideo-SAM3-X3-B7
export SC_SAM_ROOT=/Data_8TB/lht/PseudoVideo-SAM3-X3-B7/third_party/SC-SAM
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
PY=/home/violet/anaconda3/envs/mkunet_mamba/bin/python
SAM3_PY=/home/violet/anaconda3/envs/sam3/bin/python

PHASE=work/kvasir_1pct_anchors/phase1
DATA=work/kvasir_1pct_anchors/baseline_data
LABELS=work/kvasir_1pct_anchors/protocol/frozen_labeled_images.txt

echo "[1/5] export T24 predictions"
for entry in "S2 student_best.pth predictions/S2_valbest" "S2 student_final.pth predictions/S2_final" "S3 student_final.pth predictions/S3_final"; do
  set -- $entry
  run=$1; ckpt=$2; out=$3
  if [[ ! -f "$PHASE/$out/EXPORT_COMPLETE" ]]; then
    "$PY" scripts/export_t25_student_predictions.py \
      --run-dir "$PHASE/students/$run" \
      --checkpoint "$PHASE/students/$run/$ckpt" \
      --output-root "$PHASE/$out"
  fi
done

echo "[2/5] committee audit -> tiers"
if [[ ! -f "$PHASE/audit/summary.json" ]]; then
  "$PY" scripts/phase1_audit_tiers.py \
    --train-metadata "$DATA/train/metadata.jsonl" \
    --labeled-list "$LABELS" \
    --quality-root work/kvasir_1pct_anchors/stage1_feature_knn_b7_ft1pct \
    --original-manifest "$PHASE/pseudo_manifest_original.jsonl" \
    --predictions "$PHASE/predictions/S2_valbest/student_predictions_train.jsonl" \
    --predictions-name S2_valbest \
    --predictions "$PHASE/predictions/S2_final/student_predictions_train.jsonl" \
    --predictions-name S2_final \
    --predictions "$PHASE/predictions/S3_final/student_predictions_train.jsonl" \
    --predictions-name S3_final \
    --output-dir "$PHASE/audit"
fi

echo "[3/5] X3 manifest"
if [[ ! -f "$PHASE/pseudo_manifest_x3.jsonl" ]]; then
  "$PY" scripts/phase1_build_x3_manifest.py \
    --original "$PHASE/pseudo_manifest_original.jsonl" \
    --tier-a "$PHASE/audit/tier_A.jsonl" \
    --tier-b "$PHASE/audit/tier_B.jsonl" \
    --output "$PHASE/pseudo_manifest_x3.jsonl"
fi

echo "[4/5] X3 training"
if [[ ! -f "$PHASE/students/X3/TRAINING_COMPLETE" ]]; then
  "$PY" scripts/run_s27_student.py \
    --data-path "$DATA" \
    --labeled-list "$LABELS" \
    --pseudo-manifest "$PHASE/pseudo_manifest_x3.jsonl" \
    --output-dir "$PHASE/students/X3" \
    --experiment X3 \
    --seed 2026 \
    --batch-size 12 --gt-bs 3 --original-bs 3 --new-bs 6 \
    --max-iterations 40000 --val-interval 200 --num-workers 4
fi

echo "[5/5] X3 export + B7 selection"
if [[ ! -f "$PHASE/predictions/X3_final/EXPORT_COMPLETE" ]]; then
  "$PY" scripts/export_t25_student_predictions.py \
    --run-dir "$PHASE/students/X3" \
    --checkpoint "$PHASE/students/X3/student_final.pth" \
    --output-root "$PHASE/predictions/X3_final"
fi

"$PY" scripts/phase1_b7_select.py \
  --quality-root work/kvasir_1pct_anchors/stage1_feature_knn_b7_ft1pct \
  --student-predictions "$PHASE/predictions/X3_final/student_predictions_test.jsonl" \
  --output-dir "$PHASE/selection"

echo "[done] post-T24"
