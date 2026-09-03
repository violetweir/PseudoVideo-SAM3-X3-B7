#!/usr/bin/env bash
# Round-2C: strictly @256 lesion-correspondence anchor audit, validation only.
set -Eeuo pipefail

cd /Data_8TB/lht/PseudoVideo-SAM3-X3-B7

PY=/home/violet/anaconda3/envs/sam3/bin/python
BASE=work/rerun_c0_256_sam3knn_s256_base
ROUND2A=work/rerun_c0_256_round2a_fixed_knn_e33
PHASE=work/rerun_c0_256_round2c_lesion_anchor_validation
QROOT=$PHASE/quality_root
MODE=round2c_all_anchors_b0
PATCHES=$PHASE/features/sam3_base_s256_validation_anchor_patches.npz
CHECKPOINT=$BASE/medsam3_lora_b0_b6_e50/e33_full_evaluation/e33_merged_video.pt
PREDICTIONS=$BASE/predictions/X3_best/student_predictions_validation.jsonl
REPORT=reproduction_reports/C0_256_round2c_lesion_correspondence_anchor_reranking.md
export CUDA_VISIBLE_DEVICES=${DEVICE:-0}

mkdir -p "$PHASE/features" "$PHASE/b7"
rm -f "$PHASE/FAILED" "$PHASE/COMPLETE"
trap 'touch "$PHASE/FAILED"' ERR

echo "[round2c] $(date '+%F %T') start validation-only base@256 + e33@256 GPU=$CUDA_VISIBLE_DEVICES"

"$PY" scripts/extract_c0_256_round2c_base_patches.py \
  --output "$PATCHES" --split validation --feature-size 256 --batch-size 8

"$PY" scripts/audit_c0_256_round2c_lesion_anchors.py prepare \
  --phase "$PHASE" --patch-cache "$PATCHES" \
  --split validation --feature-size 256 --canvas 256 --match-topk 8

echo "[round2c] $(date '+%F %T') evaluate all eight frozen anchors with SAM3-e33 canvas=256"
"$PY" scripts/eval_route_propagation_quality.py \
  --checkpoint "$CHECKPOINT" --mode "$MODE" --root "$QROOT" \
  --split validation --canvas 256 --resume

"$PY" scripts/audit_c0_256_round2c_lesion_anchors.py summarize \
  --phase "$PHASE" --split validation --feature-size 256 --canvas 256

"$PY" scripts/build_c0_256_b7_lora_manifest.py \
  --quality-root "$ROUND2A/quality_root" \
  --student-predictions "$PREDICTIONS" \
  --split validation \
  --modes sam3enc_anchor_conditioned_target_pooling sam3enc_anchor_conditioned_patch_correspondence \
  --min-bridge 0 --max-bridge 6 --min-b7 0 --canvas 256 \
  --output "$PHASE/b7/baseline_round2a_b0_b6.jsonl" \
  --summary "$PHASE/b7/baseline_round2a_b0_b6.summary.json"

VARIANTS=(
  V0_target_pooling
  V0_patch_correspondence
)
for source in round1 x3_best; do
  VARIANTS+=("V1_lesion_mean__${source}")
  for aggregate in prototype token_topk; do
    VARIANTS+=(
      "V2_forward_${aggregate}__${source}"
      "V3_reverse_${aggregate}__${source}"
      "V4_bidirectional_${aggregate}__${source}"
    )
  done
done

for variant in "${VARIANTS[@]}"; do
  "$PY" scripts/build_c0_256_b7_lora_manifest.py \
    --quality-root "$PHASE/top3_quality_root" \
    --student-predictions "$PREDICTIONS" \
    --split validation \
    --modes "round2c_${variant}_top3" \
    --min-bridge 0 --max-bridge 0 --min-b7 0 --canvas 256 \
    --output "$PHASE/b7/${variant}.jsonl" \
    --summary "$PHASE/b7/${variant}.summary.json"
done

"$PY" scripts/audit_c0_256_round2c_lesion_anchors.py report \
  --phase "$PHASE" --report "$REPORT" \
  --split validation --feature-size 256 --canvas 256

touch "$PHASE/COMPLETE"
echo "[round2c] $(date '+%F %T') complete; validation only, no test/train propagation/training"
