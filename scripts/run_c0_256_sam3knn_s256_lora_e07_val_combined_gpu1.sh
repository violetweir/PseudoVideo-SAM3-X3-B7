#!/usr/bin/env bash
set -Eeuo pipefail

cd /Data_8TB/lht/PseudoVideo-SAM3-X3-B7

PHASE=work/rerun_c0_256_sam3knn_s256_base
RUN=$PHASE/medsam3_lora_b0_b6_e50
ROUTE_SOURCE=$PHASE/stage1_feature_knn_b0_b6
OUT=$RUN/current_best_val_sam3knn_e07
QROOT=$OUT/quality_root
BASE=/Data_8TB/lht/models/modelscope/models/facebook--sam3/snapshots/master/sam3.pt
LORA=$RUN/lora_weights/epoch_7_lora_weights.pt
MERGED=$OUT/e07_merged_video.pt
PY=/home/violet/anaconda3/envs/sam3/bin/python
export CUDA_VISIBLE_DEVICES=1

MODES=(
  sam3enc_anchor_conditioned_target_pooling
  sam3enc_anchor_conditioned_patch_correspondence
)

mkdir -p "$OUT"
rm -f "$OUT/FAILED"
trap 'touch "$OUT/FAILED"' ERR

if [[ ! -f $MERGED ]]; then
  echo "[e07-val] $(date '+%F %T') merge LoRA checkpoint"
  "$PY" scripts/merge_sam3_lora_video_checkpoint.py \
    --base-checkpoint "$BASE" --lora-weights "$LORA" --output "$MERGED"
fi

for mode in "${MODES[@]}"; do
  mkdir -p "$QROOT/$mode"
  ln -sfn "$(realpath "$ROUTE_SOURCE/$mode/validation_pool0_stage1")" \
    "$QROOT/$mode/validation_pool0_stage1"
  echo "[e07-val] $(date '+%F %T') $mode"
  "$PY" scripts/eval_route_propagation_quality.py \
    --checkpoint "$MERGED" --mode "$mode" --root "$QROOT" \
    --split validation --canvas 256 --resume
done

"$PY" scripts/summarize_c0_256_bridge_metrics.py \
  --quality-root "$QROOT" --split validation --modes "${MODES[@]}" \
  --min-bridge 0 --max-bridge 6 \
  --output-json "$OUT/two_mode_combined_b0_b6_validation.json" \
  --output-tsv "$OUT/two_mode_combined_b0_b6_validation.tsv"

touch "$OUT/COMPLETE"
echo "[e07-val] $(date '+%F %T') complete"
