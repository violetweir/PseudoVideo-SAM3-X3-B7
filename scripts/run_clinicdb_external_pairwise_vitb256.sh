#!/usr/bin/env bash
set -euo pipefail

cd /Data_8TB/lht/PseudoVideo-SAM3-X3-B7

PY=/home/violet/anaconda3/envs/sam3/bin/python
PROTOCOL=work/clinicdb_external_kvasir8/protocol
ROUTES=work/clinicdb_external_kvasir8/stage1_feature_knn_vitb256
REPORT=work/clinicdb_external_kvasir8/pairwise_ranker_vitb256
CHECKPOINT=work/kvasir_1pct_anchors/video_checkpoints/lora_p491_e20_merged_video.pt
MODEL=work/kvasir_1pct_anchors/pairwise_ranker_vitb256/frozen_internal_consensus/frozen_internal_plus_consensus_lora_p491_e20.json

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/Data_8TB/lht/PseudoVideo-SAM3-X3-B7/scripts

"$PY" scripts/prepare_clinicdb_external_kvasir8_protocol.py

generate_routes () {
  local key=$1 mode=$2 dinov3=$3 knn=$4
  echo "[routes] $key start $(date '+%F %T')"
  "$PY" scripts/stage1_feature_knn_routes.py \
    --mode "$mode" \
    --dinov3-model "$dinov3" \
    --knn-feature "$knn" \
    --feature-size 256 \
    --min-bridge 3 \
    --max-bridge 6 \
    --beam-width 32 \
    --split test \
    --protocol-root "$PROTOCOL" \
    --output-root "$ROUTES"
  echo "[routes] $key done $(date '+%F %T')"
}

quality () {
  local key=$1
  echo "[quality] $key start $(date '+%F %T')"
  "$PY" scripts/eval_route_propagation_quality.py \
    --checkpoint "$CHECKPOINT" \
    --mode "$key" \
    --root "$ROUTES" \
    --split test \
    --canvas 256 \
    --tag lora_p491_e20 \
    --resume
  echo "[quality] $key done $(date '+%F %T')"
}

generate_routes t18_corrected t18_corrected vits16 patch_mean
generate_routes dino_global_pooling dino_global_pooling vitb16 patch_mean
generate_routes dino_patch_average dino_patch_average vitb16 patch_mean
for mode in anchor_conditioned_target_pooling anchor_conditioned_patch_correspondence; do
  for knn in patch_mean cls pooled cond; do
    key=$mode
    if [[ "$knn" != patch_mean ]]; then
      key="${mode}__knn_${knn}"
    fi
    generate_routes "$key" "$mode" vitb16 "$knn"
  done
done

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
  quality "$key"
done

"$PY" scripts/eval_frozen_pairwise_route_ranker_vitb256.py \
  --quality-root "$ROUTES" \
  --model "$MODEL" \
  --output-root "$REPORT" \
  --split test \
  --tag lora_p491_e20 \
  --evaluation-role frozen_external_clinicdb_test

echo "CLINICDB_EXTERNAL_PAIRWISE_DONE $(date '+%F %T')"
