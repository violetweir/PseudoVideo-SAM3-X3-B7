#!/usr/bin/env bash
# Phase-1 pre-T24: mainline-style pseudo568 selection + S3 consensus targets.
set -euo pipefail

cd /Data_8TB/lht/PseudoVideo-SAM3-X3-B7
PY=/home/violet/anaconda3/envs/sam3/bin/python
PHASE=work/kvasir_1pct_anchors/phase1

echo "[1/2] mainline-style pseudo568 selection"
"$PY" scripts/select_phase1_mainline_pseudo568.py \
  --quality-root work/kvasir_1pct_anchors/stage1_feature_knn_b7_ft1pct \
  --output "$PHASE/pseudo_manifest_original.jsonl"

echo "[2/2] S3 consensus targets"
"$PY" scripts/prepare_phase1_s3_consensus.py \
  --original-manifest "$PHASE/pseudo_manifest_original.jsonl" \
  --quality-root work/kvasir_1pct_anchors/stage1_feature_knn_b7_ft1pct \
  --output-root "$PHASE/S3_consensus"

echo "[done] pre-T24"
