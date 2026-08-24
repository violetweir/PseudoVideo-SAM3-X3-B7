#!/usr/bin/env bash
set -Eeuo pipefail

cd /Data_8TB/lht/PseudoVideo-SAM3-X3-B7

PHASE=work/rerun_c0_256_sam3knn_s256_base
ROOT=$PHASE/stage1_feature_knn_b0_b6
OUT=$PHASE/current_base_test
BASE=/Data_8TB/lht/models/modelscope/models/facebook--sam3/snapshots/master/sam3.pt
PY_SAM3=/home/violet/anaconda3/envs/sam3/bin/python
PY_STUDENT=/home/violet/anaconda3/envs/mkunet_mamba/bin/python
export CUDA_VISIBLE_DEVICES=1

MODES=(
  sam3enc_anchor_conditioned_target_pooling
  sam3enc_anchor_conditioned_patch_correspondence
)

mkdir -p "$OUT"
rm -f "$OUT/FAILED"
trap 'touch "$OUT/FAILED"' ERR

if [[ ! -f $PHASE/predictions/X3_best/student_predictions_test.jsonl ]]; then
  echo "[test] $(date '+%F %T') export X3-best test predictions"
  "$PY_STUDENT" scripts/export_t25_student_predictions.py \
    --run-dir "$PHASE/students/X3" \
    --checkpoint "$PHASE/students/X3/student_best.pth" \
    --output-root "$PHASE/predictions/X3_best" \
    --splits test
fi

for mode in "${MODES[@]}"; do
  echo "[test] $(date '+%F %T') base SAM3 $mode b0-b6"
  "$PY_SAM3" scripts/eval_route_propagation_quality.py \
    --checkpoint "$BASE" --mode "$mode" --root "$ROOT" \
    --split test --canvas 256 --resume
done

"$PY_STUDENT" scripts/summarize_c0_256_bridge_metrics.py \
  --quality-root "$ROOT" --split test --modes "${MODES[@]}" \
  --min-bridge 0 --max-bridge 6 \
  --output-json "$OUT/bridge_b0_b6_test_metrics.json" \
  --output-tsv "$OUT/bridge_b0_b6_test_metrics.tsv"

"$PY_STUDENT" scripts/build_c0_256_b7_lora_manifest.py \
  --quality-root "$ROOT" \
  --student-predictions "$PHASE/predictions/X3_best/student_predictions_test.jsonl" \
  --split test --modes "${MODES[@]}" \
  --output "$OUT/x3_best_b7_b0_b6_test.jsonl" \
  --summary "$OUT/x3_best_b7_b0_b6_test.summary.json" \
  --min-bridge 0 --max-bridge 6 --min-b7 0 --canvas 256

"$PY_STUDENT" scripts/phase1_b7_select.py \
  --quality-root "$ROOT" \
  --student-predictions "$PHASE/predictions/X3_best/student_predictions_test.jsonl" \
  --output-dir "$OUT/x3_best_b7_report" \
  --min-bridge 0 --max-bridge 6

touch "$OUT/COMPLETE"
echo "[test] $(date '+%F %T') complete"
