#!/usr/bin/env bash
# Frozen Round-2D: Forward-Token Top-3 anchors, original base@256 routes, e33 validation.
set -Eeuo pipefail

cd /Data_8TB/lht/PseudoVideo-SAM3-X3-B7

PY=/home/violet/anaconda3/envs/sam3/bin/python
BASE=work/rerun_c0_256_sam3knn_s256_base
PHASE=work/rerun_c0_256_round2d_anchor_only_full_route_validation
QROOT=$PHASE/quality_root
TOP1ROOT=$PHASE/top1_quality_root
CHECKPOINT=$BASE/medsam3_lora_b0_b6_e50/e33_full_evaluation/e33_merged_video.pt
PREDICTIONS=$BASE/predictions/X3_best/student_predictions_validation.jsonl
export CUDA_VISIBLE_DEVICES=${DEVICE:-0}

MODES=(
  sam3enc_anchor_conditioned_target_pooling
  sam3enc_anchor_conditioned_patch_correspondence
)

mkdir -p "$PHASE" "$TOP1ROOT" "$PHASE/b7"
rm -f "$PHASE/FAILED" "$PHASE/COMPLETE"
trap 'touch "$PHASE/FAILED"' ERR

for mode in "${MODES[@]}"; do
  key="round2d_forward_token_top3_${mode}"
  "$PY" scripts/generate_c0_256_round2d_forward_token_routes.py \
    --mode "$mode" --mode-key "$key" --output-root "$QROOT" \
    --feature-size 256 --k-anchors 3 --min-bridge 0 --max-bridge 6 --beam-width 32
  "$PY" scripts/seed_c0_256_round2d_e33_quality.py \
    --quality-root "$QROOT" --mode-key "$key" --round2a-mode "$mode"
done

for mode in "${MODES[@]}"; do
  key="round2d_forward_token_top3_${mode}"
  echo "[round2d] $(date '+%F %T') propagate mode=$key e33 canvas=256 GPU=$CUDA_VISIBLE_DEVICES"
  "$PY" scripts/eval_route_propagation_quality.py \
    --checkpoint "$CHECKPOINT" --mode "$key" --root "$QROOT" \
    --split validation --canvas 256 --resume
done

KEYS=(
  round2d_forward_token_top3_sam3enc_anchor_conditioned_target_pooling
  round2d_forward_token_top3_sam3enc_anchor_conditioned_patch_correspondence
)
"$PY" scripts/summarize_c0_256_round2d_anchor_only.py \
  --phase "$PHASE" --modes "${KEYS[@]}" \
  --top1-quality-root "$TOP1ROOT" --output "$PHASE/anchor_only_b0_b6_validation.json"

"$PY" scripts/build_c0_256_b7_lora_manifest.py \
  --quality-root "$TOP1ROOT" --student-predictions "$PREDICTIONS" \
  --split validation --modes "${KEYS[@]}" \
  --min-bridge 0 --max-bridge 6 --min-b7 0 --canvas 256 \
  --output "$PHASE/b7/forward_token_top1_b0_b6.jsonl" \
  --summary "$PHASE/b7/forward_token_top1_b0_b6.summary.json"

"$PY" scripts/build_c0_256_b7_lora_manifest.py \
  --quality-root "$QROOT" --student-predictions "$PREDICTIONS" \
  --split validation --modes "${KEYS[@]}" \
  --min-bridge 0 --max-bridge 6 --min-b7 0 --canvas 256 \
  --output "$PHASE/b7/forward_token_top3_b0_b6.jsonl" \
  --summary "$PHASE/b7/forward_token_top3_b0_b6.summary.json"

touch "$PHASE/COMPLETE"
echo "[round2d] $(date '+%F %T') complete; validation only, no test/train propagation/training"
