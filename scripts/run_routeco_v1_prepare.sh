#!/usr/bin/env bash
set -euo pipefail

cd /Data_8TB/lht/PseudoVideo-SAM3-X3-B7
export CUDA_VISIBLE_DEVICES=0

PY=/home/violet/anaconda3/envs/sam3/bin/python
CKPT=/Data_8TB/lht/models/modelscope/models/facebook--sam3/snapshots/master/sam3.pt
ROOT=work/kvasir_1pct_anchors/routeco_sam3_v1_routes

mkdir -p "$ROOT"
ln -sfn ../stage1_feature_knn/features "$ROOT/features"

for mode in \
  anchor_conditioned_target_pooling \
  anchor_conditioned_patch_correspondence
do
  echo "===== routeco_v1 train routes ${mode} $(date -Is) ====="
  "$PY" scripts/stage1_feature_knn_routes.py \
    --mode "$mode" \
    --split train \
    --min-bridge 3 \
    --max-bridge 6 \
    --beam-width 32 \
    --output-root "$ROOT"

  echo "===== routeco_v1 train propagation quality ${mode} $(date -Is) ====="
  "$PY" scripts/eval_route_propagation_quality.py \
    --checkpoint "$CKPT" \
    --mode "$mode" \
    --root "$ROOT" \
    --split train \
    --canvas 512 \
    --no-target-gt \
    --resume
done

  scripts/build_routeco_v1_pseudo_manifest.py

echo "ROUTECO_V1_PREPARE_DONE $(date -Is)"
