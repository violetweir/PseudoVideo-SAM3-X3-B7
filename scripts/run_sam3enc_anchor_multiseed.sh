#!/usr/bin/env bash
# Multi-seed validation of the anchor-count ablation (focus: is n=1 stably close to n=8?).
# Runs seeds {7, 123} x N {1, 2} (seed=42 already done as n1/n2).
# Fixed KNN variant: sam3enc_anchor_conditioned_target_pooling + cond, base eval only.
set -euo pipefail

cd /Data_8TB/lht/PseudoVideo-SAM3-X3-B7
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/Data_8TB/lht/sam3:/Data_8TB/lht/PseudoVideo-SAM3-X3-B7/scripts

PY=/home/violet/anaconda3/envs/sam3/bin/python
CKPT_BASE=/Data_8TB/lht/models/modelscope/models/facebook--sam3/snapshots/master/sam3.pt
ROOT=work/kvasir_1pct_anchors/stage1_feature_knn_sam3enc_s1008_anchor_ablation
PROTOCOL=work/kvasir_1pct_anchors/protocol
MODE_KEY=sam3enc_anchor_conditioned_target_pooling__knn_cond
SEEDS=${SEEDS:-"7 123"}
NS=${NS:-"1 2"}

for SEED in $SEEDS; do
  for N in $NS; do
    echo "[subset seed=$SEED n=$N] $(date '+%F %T')"
    "$PY" scripts/build_sam3enc_anchor_subsets.py \
      --n-anchors "$N" --seed "$SEED" \
      --protocol-root "$PROTOCOL" --output-root "$ROOT"
    OUT="$ROOT/n${N}_s${SEED}"
    echo "[routes seed=$SEED n=$N] $(date '+%F %T')"
    "$PY" scripts/stage1_feature_knn_routes.py \
      --mode sam3enc_anchor_conditioned_target_pooling \
      --feature-source sam3_base --feature-size 1008 \
      --knn-feature cond \
      --support-manifest "$OUT/protocol/support_manifest_n${N}.jsonl" \
      --max-bridge 6 --beam-width 32 --split test \
      --output-root "$OUT"
    echo "[eval base seed=$SEED n=$N] $(date '+%F %T')"
    "$PY" scripts/stage1_eval_routes_forward_only.py \
      --checkpoint "$CKPT_BASE" \
      --mode "$MODE_KEY" \
      --root "$OUT" \
      --canvas 256 \
      --eval-name eval_base_no_ft_b7_forward \
      --resume
  done
done

echo "MULTISEED_N1N2_DONE $(date '+%F %T')"
