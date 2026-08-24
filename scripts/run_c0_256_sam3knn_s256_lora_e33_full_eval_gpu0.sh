#!/usr/bin/env bash
set -Eeuo pipefail

cd /Data_8TB/lht/PseudoVideo-SAM3-X3-B7

PHASE=work/rerun_c0_256_sam3knn_s256_base
RUN=$PHASE/medsam3_lora_b0_b6_e50
ROUTE_SOURCE=$PHASE/stage1_feature_knn_b0_b6
OUT=$RUN/e33_full_evaluation
QROOT=$OUT/quality_root
CONFIG=configs/c0_256_sam3knn_s256_b0_b6_medsam3_lora_e50.yaml
BASE=/Data_8TB/lht/models/modelscope/models/facebook--sam3/snapshots/master/sam3.pt
LORA=$RUN/lora_weights/epoch_33_lora_weights.pt
MERGED=$OUT/e33_merged_video.pt
PY=/home/violet/anaconda3/envs/sam3/bin/python
export CUDA_VISIBLE_DEVICES=0

MODES=(
  sam3enc_anchor_conditioned_target_pooling
  sam3enc_anchor_conditioned_patch_correspondence
)

mkdir -p "$OUT"
rm -f "$OUT/COMPLETE" "$OUT/FAILED"
trap 'touch "$OUT/FAILED"' ERR

run_direct() {
  local resolution=$1
  local prompt_mode=$2
  local output=$3
  if [[ ! -s $output ]]; then
    "$PY" scripts/eval_sam3_lora_direct_split.py \
      --config "$CONFIG" --lora "$LORA" --split test \
      --effective-resolution "$resolution" --prompt-mode "$prompt_mode" \
      --output "$output"
  fi
}

echo "[e33] $(date '+%F %T') direct test r1008 text"
run_direct 1008 category "$OUT/direct_test_r1008_text.json"
echo "[e33] $(date '+%F %T') direct test effective-r256 text"
run_direct 256 category "$OUT/direct_test_effective_r256_text.json"
echo "[e33] $(date '+%F %T') direct test effective-r256 empty text"
run_direct 256 empty "$OUT/direct_test_effective_r256_empty_text.json"

if [[ ! -f $MERGED ]]; then
  echo "[e33] $(date '+%F %T') merge LoRA video checkpoint"
  "$PY" scripts/merge_sam3_lora_video_checkpoint.py \
    --base-checkpoint "$BASE" --lora-weights "$LORA" --output "$MERGED"
fi

for split in validation test; do
  for mode in "${MODES[@]}"; do
    mkdir -p "$QROOT/$mode"
    ln -sfn "$(realpath "$ROUTE_SOURCE/$mode/${split}_pool0_stage1")" \
      "$QROOT/$mode/${split}_pool0_stage1"
    echo "[e33] $(date '+%F %T') propagation $split $mode"
    "$PY" scripts/eval_route_propagation_quality.py \
      --checkpoint "$MERGED" --mode "$mode" --root "$QROOT" \
      --split "$split" --canvas 256 --resume
  done

  "$PY" scripts/summarize_c0_256_bridge_metrics.py \
    --quality-root "$QROOT" --split "$split" --modes "${MODES[@]}" \
    --min-bridge 0 --max-bridge 6 \
    --output-json "$OUT/two_mode_b0_b6_${split}.json" \
    --output-tsv "$OUT/two_mode_b0_b6_${split}.tsv"
done

touch "$OUT/COMPLETE"
echo "[e33] $(date '+%F %T') complete"
