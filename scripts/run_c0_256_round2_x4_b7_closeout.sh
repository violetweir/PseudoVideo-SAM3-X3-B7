#!/usr/bin/env bash
# Close e33/X4/B7 without launching any additional SAM3 training.
set -Eeuo pipefail

cd /Data_8TB/lht/PseudoVideo-SAM3-X3-B7

PHASE=work/rerun_c0_256_round2a_fixed_knn_e33
QROOT=$PHASE/quality_root
SAMPY=/home/violet/anaconda3/envs/sam3/bin/python
MKPY=/home/violet/anaconda3/envs/mkunet_mamba/bin/python
BASE=work/rerun_c0_256_sam3knn_s256_base
B7=$PHASE/x4_b7_closeout
LOG=$PHASE/x4_b7_closeout.log
MODES=(
  sam3enc_anchor_conditioned_target_pooling
  sam3enc_anchor_conditioned_patch_correspondence
)

mkdir -p "$B7"
rm -f "$PHASE/X4_B7_FAILED"
trap 'touch "$PHASE/X4_B7_FAILED"' ERR

export_variant() {
  local variant=$1 gpu=$2 checkpoint output
  if [[ $variant == best ]]; then
    checkpoint=$PHASE/students/X4/student_best.pth
    output=$PHASE/predictions/X4_best
  else
    checkpoint=$PHASE/students/X4/student_final.pth
    output=$PHASE/predictions/X4_final
  fi
  if [[ -f $output/EXPORT_COMPLETE ]]; then
    echo "[round2] X4 $variant predictions already exist"
    return
  fi
  CUDA_VISIBLE_DEVICES=$gpu "$MKPY" scripts/export_t25_student_predictions.py \
    --run-dir "$PHASE/students/X4" --checkpoint "$checkpoint" \
    --output-root "$output" --splits train validation test
}

echo "[round2] $(date '+%F %T') export X4 best/final on GPU0/GPU1" | tee -a "$LOG"
export_variant best 0 > "$PHASE/x4_best_export.log" 2>&1 &
best_pid=$!
export_variant final 1 > "$PHASE/x4_final_export.log" 2>&1 &
final_pid=$!
wait "$best_pid"
wait "$final_pid"

for selector in X3_best X4_best X4_final; do
  if [[ $selector == X3_best ]]; then
    predictions=$BASE/predictions/X3_best
  else
    predictions=$PHASE/predictions/$selector
  fi
  for split in validation test; do
    echo "[round2] $(date '+%F %T') B7 $selector $split" | tee -a "$LOG"
    "$SAMPY" scripts/build_c0_256_b7_lora_manifest.py \
      --quality-root "$QROOT" \
      --student-predictions "$predictions/student_predictions_${split}.jsonl" \
      --split "$split" --modes "${MODES[@]}" \
      --min-bridge 0 --max-bridge 6 --min-b7 0 --canvas 256 \
      --output "$B7/${selector}_${split}.jsonl" \
      --summary "$B7/${selector}_${split}.summary.json" \
      2>&1 | tee -a "$LOG"
  done
done

"$SAMPY" scripts/summarize_c0_256_round2_b7_closeout.py \
  --root "$B7" --output "$B7/x3_x4_e33_b7_comparison.json" \
  2>&1 | tee -a "$LOG"

touch "$PHASE/X4_B7_COMPLETE"
echo "[round2] $(date '+%F %T') X4+B7 validation/test complete; no further SAM3 training" | tee -a "$LOG"
