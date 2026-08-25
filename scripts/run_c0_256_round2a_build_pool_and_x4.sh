#!/usr/bin/env bash
# Build the validation-frozen Round-2A hard pseudo pool, then train Student-X4.
set -Eeuo pipefail

cd /Data_8TB/lht/PseudoVideo-SAM3-X3-B7

SAMPY=/home/violet/anaconda3/envs/sam3/bin/python
MKPY=/home/violet/anaconda3/envs/mkunet_mamba/bin/python
PHASE=work/rerun_c0_256_round2a_fixed_knn_e33
QROOT=$PHASE/quality_root
X3ROOT=work/rerun_c0_256_sam3knn_s256_base
DATA=work/kvasir_1pct_anchors/baseline_data
LABELS=work/kvasir_1pct_anchors/protocol/frozen_labeled_images.txt
THRESHOLD=${B7_THRESHOLD:-$(
  "$SAMPY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["recommended"]["min_b7"])' \
    "$PHASE/b7_calibration/calibration_frontier.json"
)}
OUT=$PHASE/students/X4
LOG=$PHASE/x4_train.log
MODES=(
  sam3enc_anchor_conditioned_target_pooling
  sam3enc_anchor_conditioned_patch_correspondence
)

if [[ ! -f "$PHASE/TRAIN_PROP_COMPLETE" || ! -f "$PHASE/B7_CALIBRATION_COMPLETE" ]]; then
  echo "Round-2A propagation/calibration is incomplete" >&2
  exit 2
fi

mkdir -p "$PHASE/manifests" "$OUT"
rm -f "$PHASE/X4_FAILED"
trap 'touch "$PHASE/X4_FAILED"' ERR

"$SAMPY" scripts/build_c0_256_b7_lora_manifest.py \
  --quality-root "$QROOT" \
  --student-predictions "$X3ROOT/predictions/X3_best/student_predictions_train.jsonl" \
  --split train --modes "${MODES[@]}" \
  --min-bridge 0 --max-bridge 6 --min-b7 "$THRESHOLD" --canvas 256 \
  --sample-type original \
  --output "$PHASE/manifests/x4_train_b7_raw.jsonl" \
  --summary "$PHASE/manifests/x4_train_b7_raw.summary.json" \
  2>&1 | tee -a "$LOG"

# Preserve X3's S27 stream assignment for targets already present in the X3
# manifest. Newly admitted targets enter the original hard-label stream.
"$SAMPY" scripts/prepare_round2a_x4_manifest.py \
  --b7-manifest "$PHASE/manifests/x4_train_b7_raw.jsonl" \
  --x3-reference "$X3ROOT/pseudo_manifest_x3.jsonl" \
  --output "$PHASE/manifests/x4_train_b7.jsonl" \
  --summary "$PHASE/manifests/x4_train_b7.summary.json" \
  2>&1 | tee -a "$LOG"

# Keep the exact X3 batch recipe: 3 GT + 3 original + 6 tier-A/B pseudo.
CUDA_VISIBLE_DEVICES=${DEVICE:-0} "$MKPY" scripts/run_s27_student.py \
  --data-path "$DATA" --labeled-list "$LABELS" \
  --pseudo-manifest "$PHASE/manifests/x4_train_b7.jsonl" \
  --output-dir "$OUT" --experiment X4_round2a --seed 2026 \
  --batch-size 12 --gt-bs 3 --original-bs 3 --new-bs 6 \
  --max-iterations 40000 --val-interval 200 --num-workers 4 \
  --resume \
  2>&1 | tee -a "$LOG"

touch "$PHASE/X4_TRAIN_COMPLETE"
echo "[round2a] $(date '+%F %T') X4 training complete" | tee -a "$LOG"
