#!/usr/bin/env bash
# IMR (Image-native Multimodal Retrieval) KNN @256 — Kvasir 1% protocol
# Story: same 256x256 protocol as DINOv3 ViT-B @256 and SAM3-enc @256 tables,
#        retrieval features upgraded to mask-free image-native descriptors
#        (spatial pyramid + contrast weighting) + Qwen image-text rank fusion.
# Protocol: frozen SAM3 base / lora_p491_e20 eval @canvas 256, beam 32, b0-b6.
set -euo pipefail

cd /Data_8TB/lht/PseudoVideo-SAM3-X3-B7
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/Data_8TB/lht/sam3:/Data_8TB/lht/PseudoVideo-SAM3-X3-B7/scripts

PY=/home/violet/anaconda3/envs/sam3/bin/python
CKPT_BASE=/Data_8TB/lht/models/modelscope/models/facebook--sam3/snapshots/master/sam3.pt
CKPT_LORA=work/kvasir_1pct_anchors/video_checkpoints/lora_p491_e20_merged_video.pt
ROOT=work/kvasir_1pct_anchors/stage1_feature_knn_sam3enc_imr_s256
QWEN_ROOT=work/kvasir_1pct_anchors/qwen35_image_descriptions
LAMBDAS=("0.0" "0.25" "0.5" "0.75" "1.0")
mkdir -p "$ROOT"

gen_route () {
  local split=$1 mode=$2 lambda=$3
  local extra=()
  if [ "$mode" = "sam3enc_imr" ]; then
    extra=(--qwen-desc-root "$QWEN_ROOT" --text-blend "$lambda")
  fi
  echo "[routes] $mode lambda=$lambda split=$split start $(date '+%F %T')"
  "$PY" scripts/stage1_feature_knn_routes.py \
    --mode "$mode" \
    --feature-source sam3_base \
    --feature-size 256 \
    --max-bridge 6 \
    --beam-width 32 \
    --split "$split" \
    --output-root "$ROOT" \
    "${extra[@]}"
  echo "[routes] $mode lambda=$lambda split=$split done $(date '+%F %T')"
}

eval_routes () {
  local key=$1 ckpt=$2 eval_name=$3
  echo "[eval] $key $eval_name start $(date '+%F %T')"
  "$PY" scripts/stage1_eval_routes_forward_only.py \
    --checkpoint "$ckpt" \
    --mode "$key" \
    --root "$ROOT" \
    --canvas 256 \
    --eval-name "$eval_name" \
    --resume
  echo "[eval] $key $eval_name done $(date '+%F %T')"
}

# ---- Phase 0a: Qwen image descriptions (skip with QWEN_SKIP=1) ----
if [ "${QWEN_SKIP:-0}" != "1" ]; then
  mkdir -p "$QWEN_ROOT"
  "$PY" scripts/qwen35_describe_images.py \
    --manifest work/kvasir_1pct_anchors/protocol/merged_manifest.jsonl \
    --output "$QWEN_ROOT/qwen35_image_descriptions.jsonl"
  "$PY" scripts/qwen35_describe_images.py \
    --manifest work/kvasir_1pct_anchors/protocol/support_manifest.jsonl \
    --overlay-gt-mask \
    --output "$QWEN_ROOT/qwen35_image_descriptions_overlay.jsonl"
fi

# ---- Phase 1: validation routes (first run triggers IMR feature extraction) ----
gen_route validation sam3enc_pyramid 0.0
gen_route validation sam3enc_contrast 0.0
gen_route validation sam3enc_pyramid_contrast 0.0
for L in "${LAMBDAS[@]}"; do
  gen_route validation sam3enc_imr "$L"
done

# ---- Phase 2: validation eval (base) ----
for key in \
  sam3enc_pyramid \
  sam3enc_contrast \
  sam3enc_pyramid_contrast \
  sam3enc_imr \
  sam3enc_imr__tb025 \
  sam3enc_imr__tb050 \
  sam3enc_imr__tb075 \
  sam3enc_imr__tb100
do
  eval_routes "$key" "$CKPT_BASE" eval_base_no_ft_b7_forward
done

echo "IMR_VALIDATION_DONE $(date '+%F %T')"
echo ">> Pick the best lambda on validation, then set BEST_LAMBDA below and re-run for test."

# ---- Phase 3: test routes + eval (only when BEST_LAMBDA is set) ----
if [ -n "${BEST_LAMBDA:-}" ]; then
  gen_route test sam3enc_pyramid 0.0
  gen_route test sam3enc_contrast 0.0
  gen_route test sam3enc_pyramid_contrast 0.0
  gen_route test sam3enc_imr "$BEST_LAMBDA"

  IMR_KEY=sam3enc_imr
  if [ "$BEST_LAMBDA" != "0.0" ]; then
    IMR_KEY="sam3enc_imr__tb$(python3 -c "print(f'{int(round(float('$BEST_LAMBDA')*100)):03d}')")"
  fi
  for key in sam3enc_pyramid sam3enc_contrast sam3enc_pyramid_contrast "$IMR_KEY"; do
    eval_routes "$key" "$CKPT_BASE" eval_base_no_ft_b7_forward
    eval_routes "$key" "$CKPT_LORA" eval_lora_p491_e20
  done

  "$PY" scripts/summarize_stage1_knn_vitb256.py --root "$ROOT" \
    --eval-name eval_base_no_ft_b7_forward --label imr_s256_base \
    --title "IMR KNN @256 features (SAM3-enc pyramid+contrast+Qwen text), SAM3 base @256 eval" --no-old-reference
  "$PY" scripts/summarize_stage1_knn_vitb256.py --root "$ROOT" \
    --eval-name eval_lora_p491_e20 --label imr_s256_lora \
    --title "IMR KNN @256 features (SAM3-enc pyramid+contrast+Qwen text), lora_p491_e20 @256 eval" --no-old-reference
  echo "IMR_TEST_DONE $(date '+%F %T')"
fi
