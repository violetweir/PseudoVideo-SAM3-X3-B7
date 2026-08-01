#!/usr/bin/env bash
set -euo pipefail

cd /Data_8TB/lht/PseudoVideo-SAM3-X3-B7
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

SAM3_PY=/home/violet/anaconda3/envs/sam3/bin/python
MANIFEST=work/kvasir_1pct_anchors/protocol/merged_manifest.jsonl
SUPPORT=work/kvasir_1pct_anchors/protocol/support_manifest.jsonl
OUT_BASE=work/kvasir_1pct_anchors/model_routes_max6
mkdir -p "$OUT_BASE"

declare -A CKPTS
CKPTS[base_no_ft]=/Data_8TB/lht/models/modelscope/models/facebook--sam3/snapshots/master/sam3.pt
CKPTS[ft_1pct]=work/kvasir_1pct_anchors/video_checkpoints/ft_1pct_merged_video.pt
CKPTS[ft_5pct]=work/kvasir_1pct_anchors/video_checkpoints/ft_5pct_merged_video.pt
CKPTS[ft_10pct]=work/kvasir_1pct_anchors/video_checkpoints/ft_10pct_merged_video.pt
CKPTS[ft_20pct_latest_after_crash]=work/kvasir_1pct_anchors/video_checkpoints/ft_20pct_latest_after_crash_merged_video.pt

for name in base_no_ft ft_1pct ft_5pct ft_10pct ft_20pct_latest_after_crash; do
  echo "===== max6 $name ====="
  "$SAM3_PY" scripts/run_kvasir_max6_pseudovideo.py \
    --manifest "$MANIFEST" \
    --support-manifest "$SUPPORT" \
    --output-root "$OUT_BASE/$name" \
    --checkpoint "${CKPTS[$name]}" \
    --resume
  echo "===== done max6 $name ====="
done
