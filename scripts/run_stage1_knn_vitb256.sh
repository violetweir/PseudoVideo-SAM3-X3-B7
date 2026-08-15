#!/usr/bin/env bash
set -euo pipefail

cd /Data_8TB/lht/PseudoVideo-SAM3-X3-B7
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/Data_8TB/lht/PseudoVideo-SAM3-X3-B7/scripts

PY=/home/violet/anaconda3/envs/sam3/bin/python
ROOT=work/kvasir_1pct_anchors/stage1_feature_knn_vitb256
CKPT=/Data_8TB/lht/models/modelscope/models/facebook--sam3/snapshots/master/sam3.pt
mkdir -p "$ROOT"

gen_route () {
  local key=$1 mode=$2 dinov3=$3 knn=$4
  echo "[routes] $key start $(date '+%F %T')"
  "$PY" scripts/stage1_feature_knn_routes.py \
    --mode "$mode" \
    --dinov3-model "$dinov3" \
    --knn-feature "$knn" \
    --feature-size 256 \
    --max-bridge 6 \
    --beam-width 32 \
    --split test \
    --output-root "$ROOT"
  echo "[routes] $key done $(date '+%F %T')"
}

eval_routes () {
  local key=$1
  echo "[eval] $key start $(date '+%F %T')"
  "$PY" scripts/stage1_eval_routes_forward_only.py \
    --checkpoint "$CKPT" \
    --mode "$key" \
    --root "$ROOT" \
    --canvas 256 \
    --resume
  echo "[eval] $key done $(date '+%F %T')"
}

# ---- Phase 1: routes ----
# Serial first: t18 descriptors + DINOv3 ViT-B @256 feature extraction
gen_route t18_corrected t18_corrected vits16 patch_mean
gen_route dino_global_pooling dino_global_pooling vitb16 patch_mean

# Remaining route jobs in parallel (CPU-bound, shared feature npz already extracted)
PIDS=()
gen_route dino_patch_average dino_patch_average vitb16 patch_mean & PIDS+=($!)

for mode in anchor_conditioned_target_pooling anchor_conditioned_patch_correspondence; do
  for knn in patch_mean cls pooled cond; do
    key=$mode
    if [ "$knn" != "patch_mean" ]; then
      key="${mode}__knn_${knn}"
    fi
    gen_route "$key" "$mode" vitb16 "$knn" & PIDS+=($!)
  done
done

wait "${PIDS[@]}"
echo "ROUTES_VITB256_DONE"

# ---- Phase 2: SAM3 forward-only eval @256 ----
for key in \
  t18_corrected \
  dino_global_pooling \
  dino_patch_average \
  anchor_conditioned_target_pooling \
  anchor_conditioned_target_pooling__knn_cls \
  anchor_conditioned_target_pooling__knn_pooled \
  anchor_conditioned_target_pooling__knn_cond \
  anchor_conditioned_patch_correspondence \
  anchor_conditioned_patch_correspondence__knn_cls \
  anchor_conditioned_patch_correspondence__knn_pooled \
  anchor_conditioned_patch_correspondence__knn_cond
do
  eval_routes "$key"
done

# ---- Phase 3: summary ----
"$PY" scripts/summarize_stage1_knn_vitb256.py --root "$ROOT"

echo "ALL_VITB256_DONE"
