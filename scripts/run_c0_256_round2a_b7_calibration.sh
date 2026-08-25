#!/usr/bin/env bash
# Validation-only B7 recalibration for Round-2A using frozen X3-best q_model.
set -Eeuo pipefail

cd /Data_8TB/lht/PseudoVideo-SAM3-X3-B7

PY=/home/violet/anaconda3/envs/sam3/bin/python
PHASE=work/rerun_c0_256_round2a_fixed_knn_e33
QROOT=$PHASE/quality_root
ROUTES=work/rerun_c0_256_sam3knn_s256_base/stage1_feature_knn_b0_b6
E33=work/rerun_c0_256_sam3knn_s256_base/medsam3_lora_b0_b6_e50/e33_full_evaluation
PRED=work/rerun_c0_256_sam3knn_s256_base/predictions/X3_best/student_predictions_validation.jsonl
OUT=$PHASE/b7_calibration
LOG=$PHASE/b7_calibration.log
MODES=(
  sam3enc_anchor_conditioned_target_pooling
  sam3enc_anchor_conditioned_patch_correspondence
)

mkdir -p "$OUT"
rm -f "$PHASE/B7_CALIBRATION_FAILED"
trap 'touch "$PHASE/B7_CALIBRATION_FAILED"' ERR

# Calibration can run while train propagation is still processing the first
# mode, so establish both frozen-route and existing-validation links here too.
for mode in "${MODES[@]}"; do
  mkdir -p "$QROOT/$mode"
  ln -sfn "$(realpath "$ROUTES/$mode/validation_pool0_stage1")" \
    "$QROOT/$mode/validation_pool0_stage1"
  ln -sfn "$(realpath "$E33/quality_root/$mode/propagation_quality_validation")" \
    "$QROOT/$mode/propagation_quality_validation"
done

for threshold in 0.00 0.90 0.92 0.94 0.96 0.97 0.98; do
  tag=${threshold/./}
  extra=()
  if [[ $threshold == 0.00 ]]; then
    extra=(--all-candidates-output "$OUT/validation_all_candidates.jsonl")
  fi
  "$PY" scripts/build_c0_256_b7_lora_manifest.py \
    --quality-root "$QROOT" --student-predictions "$PRED" \
    --split validation --modes "${MODES[@]}" \
    --min-bridge 0 --max-bridge 6 --min-b7 "$threshold" --canvas 256 \
    --output "$OUT/validation_min${tag}.jsonl" \
    --summary "$OUT/validation_min${tag}.summary.json" \
    "${extra[@]}" \
    2>&1 | tee -a "$LOG"
done

"$PY" scripts/summarize_round2a_b7_calibration.py \
  --manifest "$OUT/validation_min000.jsonl" \
  --target-dice 0.95 \
  --output-json "$OUT/calibration_frontier.json" \
  --output-tsv "$OUT/calibration_frontier.tsv" \
  2>&1 | tee -a "$LOG"

touch "$PHASE/B7_CALIBRATION_COMPLETE"
echo "[round2a] $(date '+%F %T') validation B7 calibration complete" | tee -a "$LOG"
