#!/usr/bin/env bash
set -euo pipefail

cd /Data_8TB/lht/PseudoVideo-SAM3-X3-B7
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/Data_8TB/lht/sam3:/Data_8TB/lht/PseudoVideo-SAM3-X3-B7/scripts

PY=/home/violet/anaconda3/envs/sam3/bin/python
CKPT_BASE=/Data_8TB/lht/models/modelscope/models/facebook--sam3/snapshots/master/sam3.pt
CKPT_LORA=work/kvasir_1pct_anchors/video_checkpoints/lora_p491_e20_merged_video.pt
ROOT1008=work/kvasir_1pct_anchors/stage1_feature_knn_sam3enc_s1008
ROOT256=work/kvasir_1pct_anchors/stage1_feature_knn_sam3enc_s256

gen_route () {
  local root=$1 size=$2 mode=$3 knn=$4
  local key=$mode
  if [ "$knn" != "patch_mean" ]; then
    key="${mode}__knn_${knn}"
  fi
  echo "[routes] $key s${size} start $(date '+%F %T')"
  "$PY" scripts/stage1_feature_knn_routes.py \
    --mode "$mode" \
    --feature-source sam3_base \
    --feature-size "$size" \
    --knn-feature "$knn" \
    --max-bridge 6 \
    --beam-width 32 \
    --split test \
    --output-root "$root"
  echo "[routes] $key s${size} done $(date '+%F %T')"
}

eval_routes () {
  local root=$1 key=$2 ckpt=$3 eval_name=$4
  echo "[eval] $key $eval_name start $(date '+%F %T')"
  "$PY" scripts/stage1_eval_routes_forward_only.py \
    --checkpoint "$ckpt" \
    --mode "$key" \
    --root "$root" \
    --canvas 256 \
    --eval-name "$eval_name" \
    --resume
  echo "[eval] $key $eval_name done $(date '+%F %T')"
}

# ---- Phase 1: routes (serial feature extraction first, then parallel) ----
gen_route "$ROOT1008" 1008 sam3enc_patch_average patch_mean
gen_route "$ROOT256" 256 sam3enc_patch_average patch_mean

PIDS=()
for root_size in "1008 $ROOT1008" "256 $ROOT256"; do
  set -- $root_size
  size=$1
  root=$2
  for mode in sam3enc_anchor_conditioned_target_pooling sam3enc_anchor_conditioned_patch_correspondence; do
    for knn in patch_mean pooled cond; do
      gen_route "$root" "$size" "$mode" "$knn" & PIDS+=($!)
    done
  done
done
wait "${PIDS[@]}"
echo "SAM3ENC_ROUTES_DONE $(date '+%F %T')"

# ---- Phase 2: SAM3 @256 forward-only eval (base + lora_p491_e20) ----
KEYS=(
  sam3enc_patch_average
  sam3enc_anchor_conditioned_target_pooling
  sam3enc_anchor_conditioned_target_pooling__knn_pooled
  sam3enc_anchor_conditioned_target_pooling__knn_cond
  sam3enc_anchor_conditioned_patch_correspondence
  sam3enc_anchor_conditioned_patch_correspondence__knn_pooled
  sam3enc_anchor_conditioned_patch_correspondence__knn_cond
)
for root in "$ROOT1008" "$ROOT256"; do
  for key in "${KEYS[@]}"; do
    eval_routes "$root" "$key" "$CKPT_BASE" eval_base_no_ft_b7_forward
  done
  for key in "${KEYS[@]}"; do
    eval_routes "$root" "$key" "$CKPT_LORA" eval_lora_p491_e20
  done
done

# ---- Phase 3: summaries ----
"$PY" scripts/summarize_stage1_knn_vitb256.py --root "$ROOT1008" --eval-name eval_base_no_ft_b7_forward --label sam3enc_s1008_base --title "SAM3 encoder KNN @1008 features, SAM3 base @256 eval" --no-old-reference
"$PY" scripts/summarize_stage1_knn_vitb256.py --root "$ROOT1008" --eval-name eval_lora_p491_e20 --label sam3enc_s1008_lora --title "SAM3 encoder KNN @1008 features, lora_p491_e20 @256 eval" --no-old-reference
"$PY" scripts/summarize_stage1_knn_vitb256.py --root "$ROOT256" --eval-name eval_base_no_ft_b7_forward --label sam3enc_s256_base --title "SAM3 encoder KNN @256 features, SAM3 base @256 eval" --no-old-reference
"$PY" scripts/summarize_stage1_knn_vitb256.py --root "$ROOT256" --eval-name eval_lora_p491_e20 --label sam3enc_s256_lora --title "SAM3 encoder KNN @256 features, lora_p491_e20 @256 eval" --no-old-reference

echo "SAM3ENC_ALL_DONE $(date '+%F %T')"
