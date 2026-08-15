#!/usr/bin/env bash
set -euo pipefail

cd /Data_8TB/lht/PseudoVideo-SAM3-X3-B7
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/Data_8TB/lht/PseudoVideo-SAM3-X3-B7/scripts

PY=/home/violet/anaconda3/envs/sam3/bin/python
ROOT=work/kvasir_1pct_anchors/stage1_feature_knn_vitb256
LORA_CKPT=work/kvasir_1pct_anchors/video_checkpoints/lora_p491_e20_merged_video.pt
BASE_CKPT=/Data_8TB/lht/models/modelscope/models/facebook--sam3/snapshots/master/sam3.pt

gen_validation () {
  local key=$1 mode=$2 dinov3=$3 knn=$4
  echo "[val-routes] $key start $(date '+%F %T')"
  "$PY" scripts/stage1_feature_knn_routes.py \
    --mode "$mode" \
    --dinov3-model "$dinov3" \
    --knn-feature "$knn" \
    --feature-size 256 \
    --max-bridge 6 \
    --beam-width 32 \
    --split validation \
    --output-root "$ROOT"
  echo "[val-routes] $key done $(date '+%F %T')"
}

pq_routes () {
  local key=$1 split=$2 ckpt=$3 tag=$4
  echo "[pq] $key $split ($tag) start $(date '+%F %T')"
  "$PY" scripts/eval_route_propagation_quality.py \
    --checkpoint "$ckpt" \
    --mode "$key" \
    --root "$ROOT" \
    --split "$split" \
    --canvas 256 \
    --tag "$tag" \
    --resume
  echo "[pq] $key $split ($tag) done $(date '+%F %T')"
}

# ---- Phase 1: validation routes (parallel CPU) ----
gen_validation t18_corrected t18_corrected vits16 patch_mean
gen_validation dino_global_pooling dino_global_pooling vitb16 patch_mean
PIDS=()
gen_validation dino_patch_average dino_patch_average vitb16 patch_mean & PIDS+=($!)
for mode in anchor_conditioned_target_pooling anchor_conditioned_patch_correspondence; do
  for knn in patch_mean cls pooled cond; do
    key=$mode
    if [ "$knn" != "patch_mean" ]; then
      key="${mode}__knn_${knn}"
    fi
    gen_validation "$key" "$mode" vitb16 "$knn" & PIDS+=($!)
  done
done
wait "${PIDS[@]}"
echo "VALIDATION_ROUTES_DONE $(date '+%F %T')"

# ---- Phase 2: propagation quality @256, validation + test, per checkpoint ----
run_pq_phase () {
  local ckpt=$1 tag=$2
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
    pq_routes "$key" validation "$ckpt" "$tag"
    pq_routes "$key" test "$ckpt" "$tag"
  done
}

echo "PQ_PHASE_LORA_START $(date '+%F %T')"
run_pq_phase "$LORA_CKPT" lora_p491_e20
echo "PQ_PHASE_BASE_START $(date '+%F %T')"
run_pq_phase "$BASE_CKPT" base

# ---- Phase 3: selector analysis (M1-M4), per checkpoint ----
"$PY" scripts/analyze_route_selector_vitb256.py --root "$ROOT" --tag lora_p491_e20
"$PY" scripts/analyze_route_selector_vitb256.py --root "$ROOT" --tag base

echo "ROUTE_SELECTOR_VITB256_DONE $(date '+%F %T')"
