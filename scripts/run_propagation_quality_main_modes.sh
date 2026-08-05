#!/usr/bin/env bash
set -euo pipefail

cd /Data_8TB/lht/PseudoVideo-SAM3-X3-B7
export CUDA_VISIBLE_DEVICES=0

PY=/home/violet/anaconda3/envs/sam3/bin/python
CKPT=/Data_8TB/lht/models/modelscope/models/facebook--sam3/snapshots/master/sam3.pt

for split in validation test
do
  if [[ "$split" == "validation" ]]; then
    ROOT=work/kvasir_1pct_anchors/stage1_feature_knn_b7_validation
  else
    ROOT=work/kvasir_1pct_anchors/stage1_feature_knn_b7
  fi
  for mode in \
    anchor_conditioned_target_pooling \
    anchor_conditioned_patch_correspondence
  do
    echo "===== propagation_quality ${split} ${mode} $(date -Is) ====="
    "$PY" scripts/eval_route_propagation_quality.py \
      --checkpoint "$CKPT" \
      --mode "$mode" \
      --root "$ROOT" \
      --split "$split" \
      --canvas 512 \
      --resume
  done
done

echo "PROPAGATION_QUALITY_MAIN_MODES_DONE $(date -Is)"
