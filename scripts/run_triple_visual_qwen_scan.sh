#!/usr/bin/env bash
set -euo pipefail

cd /Data_8TB/lht/PseudoVideo-SAM3-X3-B7
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/Data_8TB/lht/sam3:/Data_8TB/lht/PseudoVideo-SAM3-X3-B7/scripts

PY=/home/violet/anaconda3/envs/sam3/bin/python
CKPT=/Data_8TB/lht/models/modelscope/models/facebook--sam3/snapshots/master/sam3.pt
ORIG=work/kvasir_1pct_anchors/stage1_feature_knn_sam3enc_s1008/features/sam3_base_s1008_features.npz
MASK=work/kvasir_1pct_anchors/stage1_feature_knn_sam3enc_s1008/features/sam3enc_mask_descriptors_s1008.npz
FRACTIONS=("0.0" "0.25" "0.5" "0.75" "1.0")

for F in "${FRACTIONS[@]}"; do
  OUTROOT="work/kvasir_1pct_anchors/triple_visual_qwen_scan_${F}"
  mkdir -p "$OUTROOT/features"
  cp "$ORIG" "$OUTROOT/features/sam3_base_s1008_features.npz"
  cp "$MASK" "$OUTROOT/features/sam3enc_mask_descriptors_s1008.npz"

  echo "[mask_fraction=$F] routes $(date '+%F %T')"
  "$PY" scripts/stage1_feature_knn_routes.py \
    --mode sam3enc_mask_visual \
    --feature-size 1008 \
    --max-bridge 6 \
    --beam-width 32 \
    --split validation \
    --text-blend 0.1 \
    --mask-visual-fraction "$F" \
    --output-root "$OUTROOT"

  echo "[mask_fraction=$F] eval $(date '+%F %T')"
  "$PY" scripts/stage1_eval_routes_forward_only.py \
    --checkpoint "$CKPT" \
    --mode sam3enc_mask_visual \
    --root "$OUTROOT" \
    --split validation \
    --canvas 256 \
    --eval-name "eval_base_no_ft_b7_forward" \
    --resume
done

echo "TRIPLE_VISUAL_QWEN_SCAN_DONE $(date '+%F %T')"
