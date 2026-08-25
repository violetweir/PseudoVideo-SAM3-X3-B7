#!/usr/bin/env bash
# User-priority Round-2B: e33 topology + e33 teacher, test b0-b6 only.
set -Eeuo pipefail

cd /Data_8TB/lht/PseudoVideo-SAM3-X3-B7

PY=/home/violet/anaconda3/envs/sam3/bin/python
BASE=work/rerun_c0_256_sam3knn_s256_base
ROUND2A=work/rerun_c0_256_round2a_fixed_knn_e33
PHASE=work/rerun_c0_256_round2b_e33_topology_factorial
ROUTES=$PHASE/stage1_feature_knn_e33_b0_b6
TEACHER=$PHASE/teachers/e33
QROOT=$TEACHER/quality_root
OUT=$PHASE/e33_test_priority
CHECKPOINT=$BASE/medsam3_lora_b0_b6_e50/e33_full_evaluation/e33_merged_video.pt
REPORT=work/reproduction_reports/C0_256_round2b_e33_knn_test_b0_b6.md
LOG=$OUT/run.log
export CUDA_VISIBLE_DEVICES=${DEVICE:-0}

MODES=(
  sam3enc_anchor_conditioned_target_pooling
  sam3enc_anchor_conditioned_patch_correspondence
)

mkdir -p "$OUT" "$TEACHER"
rm -f "$OUT/FAILED"
trap 'touch "$OUT/FAILED"' ERR

for mode in "${MODES[@]}"; do
  mkdir -p "$QROOT/$mode"
  ln -sfn "$(realpath "$ROUTES/$mode/test_pool0_stage1")" \
    "$QROOT/$mode/test_pool0_stage1"
  "$PY" scripts/audit_c0_256_round2b_topology.py seed \
    --route-root "$ROUTES" --quality-root "$QROOT" \
    --source-roots "$ROUND2A/quality_root" --mode "$mode" --split test \
    2>&1 | tee -a "$LOG"
  echo "[round2b-e33-test] $(date '+%F %T') teacher=e33 split=test mode=$mode GPU=$CUDA_VISIBLE_DEVICES" | tee -a "$LOG"
  "$PY" scripts/eval_route_propagation_quality.py \
    --checkpoint "$CHECKPOINT" --mode "$mode" --root "$QROOT" \
    --split test --canvas 256 --resume 2>&1 | tee -a "$LOG"
done

"$PY" scripts/summarize_c0_256_bridge_metrics.py \
  --quality-root "$QROOT" --split test --modes "${MODES[@]}" \
  --min-bridge 0 --max-bridge 6 \
  --output-json "$TEACHER/two_mode_b0_b6_test.json" \
  --output-tsv "$TEACHER/two_mode_b0_b6_test.tsv" \
  2>&1 | tee -a "$LOG"

"$PY" scripts/summarize_c0_256_round2b_e33_test_bridges.py \
  --baseline "$ROUND2A/two_mode_b0_b6_test.json" \
  --updated "$TEACHER/two_mode_b0_b6_test.json" \
  --output-json "$OUT/base_knn_vs_e33_knn_test_b0_b6.json" \
  --output-tsv "$OUT/base_knn_vs_e33_knn_test_b0_b6.tsv" \
  --report "$REPORT" 2>&1 | tee -a "$LOG"

touch "$TEACHER/TEST_COMPLETE" "$OUT/COMPLETE"
echo "[round2b-e33-test] $(date '+%F %T') complete; b0-b6 only, no B7" | tee -a "$LOG"
