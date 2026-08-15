#!/usr/bin/env bash
set -euo pipefail

cd /Data_8TB/lht/PseudoVideo-SAM3-X3-B7
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/Data_8TB/lht/sam3:/Data_8TB/lht/PseudoVideo-SAM3-X3-B7/scripts

PY=/home/violet/anaconda3/envs/sam3/bin/python
CKPT=/Data_8TB/lht/models/modelscope/models/facebook--sam3/snapshots/master/sam3.pt
ROOT=work/kvasir_1pct_anchors/stage1_feature_knn_sam3enc_s1008
LAMBDAS=("0.0" "0.1" "0.2" "0.5" "1.0")
MASK_DESC="$ROOT/features/sam3enc_mask_descriptors_s1008.npz"

for L in "${LAMBDAS[@]}"; do
  OUTROOT="work/kvasir_1pct_anchors/mask_visual_lambda_scan_${L}"
  mkdir -p "$OUTROOT/features"
  cp "$MASK_DESC" "$OUTROOT/features/sam3enc_mask_descriptors_s1008.npz"

  echo "[lambda=$L] routes $(date '+%F %T')"
  "$PY" scripts/stage1_feature_knn_routes.py \
    --mode sam3enc_mask_visual \
    --feature-size 1008 \
    --max-bridge 6 \
    --beam-width 32 \
    --split validation \
    --text-blend "$L" \
    --output-root "$OUTROOT"

  echo "[lambda=$L] eval $(date '+%F %T')"
  "$PY" scripts/stage1_eval_routes_forward_only.py \
    --checkpoint "$CKPT" \
    --mode sam3enc_mask_visual \
    --root "$OUTROOT" \
    --split validation \
    --canvas 256 \
    --eval-name "eval_base_no_ft_b7_forward" \
    --resume
done

echo "MASK_VISUAL_LAMBDA_SCAN_DONE $(date '+%F %T')"
