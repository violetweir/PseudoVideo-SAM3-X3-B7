#!/usr/bin/env bash
# Phase-1 Stage 2-4 runner: committee audit -> X3 manifest -> X3 training -> B7.
# Expects Stage-1 students under work/kvasir_1pct_anchors/phase1/students/S1_*.
set -euo pipefail

cd /Data_8TB/lht/PseudoVideo-SAM3-X3-B7
export SC_SAM_ROOT=/Data_8TB/lht/PseudoVideo-SAM3-X3-B7/third_party/SC-SAM
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
PY=/home/violet/anaconda3/envs/mkunet_mamba/bin/python

PHASE=work/kvasir_1pct_anchors/phase1
STUDENT_DIRS=($PHASE/students/S1_*)
echo "students: ${STUDENT_DIRS[*]}"

PRED_ARGS=()
PRED_NAME_ARGS=()
for dir in "${STUDENT_DIRS[@]}"; do
  name=$(basename "$dir")
  out=$PHASE/predictions/$name
  if [[ ! -f "$out/EXPORT_COMPLETE" ]]; then
    "$PY" scripts/export_t25_student_predictions.py \
      --run-dir "$dir" \
      --checkpoint "$dir/student_final.pth" \
      --output-root "$out"
  fi
  PRED_ARGS+=(--predictions "$out/student_predictions_train.jsonl")
  PRED_NAME_ARGS+=(--predictions-name "$name")
done

if [[ ! -f "$PHASE/audit/summary.json" ]]; then
  "$PY" scripts/phase1_audit_tiers.py \
    --train-metadata work/kvasir_1pct_anchors/baseline_data/train/metadata.jsonl \
    --route-manifest work/kvasir_1pct_anchors/routeco_sam3_v1/routeco_v1_pseudo_manifest.jsonl \
    --route-root work/kvasir_1pct_anchors/routeco_sam3_v1_routes \
    --original-manifest "$PHASE/pseudo_manifest_original.jsonl" \
    "${PRED_ARGS[@]}" "${PRED_NAME_ARGS[@]}" \
    --output-dir "$PHASE/audit"
fi

if [[ ! -f "$PHASE/pseudo_manifest_x3.jsonl" ]]; then
  "$PY" scripts/phase1_build_x3_manifest.py \
    --original "$PHASE/pseudo_manifest_original.jsonl" \
    --tier-a "$PHASE/audit/tier_A.jsonl" \
    --tier-b "$PHASE/audit/tier_B.jsonl" \
    --output "$PHASE/pseudo_manifest_x3.jsonl"
fi

if [[ ! -f "$PHASE/students/X3/TRAINING_COMPLETE" ]]; then
  "$PY" scripts/run_s27_student.py \
    --data-path work/kvasir_1pct_anchors/baseline_data \
    --labeled-list work/kvasir_1pct_anchors/protocol/frozen_labeled_images.txt \
    --pseudo-manifest "$PHASE/pseudo_manifest_x3.jsonl" \
    --output-dir "$PHASE/students/X3" \
    --experiment X3 \
    --seed 2026 \
    --batch-size 12 --gt-bs 3 --original-bs 3 --new-bs 6 \
    --max-iterations 40000 --val-interval 200 --num-workers 4
fi

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

echo "[done] phase1 stage2-4"
