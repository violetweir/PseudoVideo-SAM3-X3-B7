#!/usr/bin/env bash
# SAM3-enc @1008 anchor-count ablation: 8 -> 6/4/3/2/1 GT anchors.
# Fixed KNN variant: sam3enc_anchor_conditioned_target_pooling + --knn-feature cond.
# Every N writes anchor_selection.json (seed, ordered ids, selected/excluded) +
# a nested support manifest, so each run is reproducible.
set -euo pipefail

cd /Data_8TB/lht/PseudoVideo-SAM3-X3-B7
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/Data_8TB/lht/sam3:/Data_8TB/lht/PseudoVideo-SAM3-X3-B7/scripts

PY=/home/violet/anaconda3/envs/sam3/bin/python
CKPT_BASE=/Data_8TB/lht/models/modelscope/models/facebook--sam3/snapshots/master/sam3.pt
CKPT_LORA=work/kvasir_1pct_anchors/video_checkpoints/lora_p491_e20_merged_video.pt
ROOT=work/kvasir_1pct_anchors/stage1_feature_knn_sam3enc_s1008_anchor_ablation
PROTOCOL=work/kvasir_1pct_anchors/protocol
SEED=${SEED:-42}
NS=${NS:-"6 4 3 2 1"}
mkdir -p "$ROOT"

# Phase 0: deterministic anchor subset + per-anchor cond recompute (needs patch cache)
for N in $NS; do
  echo "[subset n=$N] $(date '+%F %T')"
  "$PY" scripts/build_sam3enc_anchor_subsets.py \
    --n-anchors "$N" --seed "$SEED" \
    --protocol-root "$PROTOCOL" --output-root "$ROOT"
done

# Phase 1: test routes (cond KNN, 1008 features)
for N in $NS; do
  echo "[routes n=$N] $(date '+%F %T')"
  "$PY" scripts/stage1_feature_knn_routes.py \
    --mode sam3enc_anchor_conditioned_target_pooling \
    --feature-source sam3_base --feature-size 1008 \
    --knn-feature cond \
    --support-manifest "$ROOT/n$N/protocol/support_manifest_n$N.jsonl" \
    --max-bridge 6 --beam-width 32 --split test \
    --output-root "$ROOT/n$N"
done

# Phase 2: eval (base, then lora) @canvas 256
# mode_key = sam3enc_anchor_conditioned_target_pooling__knn_cond (knn_feature != patch_mean)
MODE_KEY=sam3enc_anchor_conditioned_target_pooling__knn_cond
for EVAL_NAME in eval_base_no_ft_b7_forward eval_lora_p491_e20; do
  CKPT="$CKPT_BASE"
  if [ "$EVAL_NAME" = "eval_lora_p491_e20" ]; then CKPT="$CKPT_LORA"; fi
  for N in $NS; do
    echo "[eval $EVAL_NAME n=$N] $(date '+%F %T')"
    "$PY" scripts/stage1_eval_routes_forward_only.py \
      --checkpoint "$CKPT" \
      --mode "$MODE_KEY" \
      --root "$ROOT/n$N" \
      --canvas 256 \
      --eval-name "$EVAL_NAME" \
      --resume
  done
done

echo "ANCHOR_ABLATION_ALL_DONE $(date '+%F %T')"
