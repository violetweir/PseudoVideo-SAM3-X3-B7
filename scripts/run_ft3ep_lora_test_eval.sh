#!/usr/bin/env bash
# Merge one training checkpoint into the base video checkpoint and run the
# forward-only test eval (b0-b6, both route modes, per-bridge Dice, no router).
#
# Usage:
#   run_ft3ep_lora_test_eval.sh <name> ft  <full_checkpoint.pt>      <gpu>
#   run_ft3ep_lora_test_eval.sh <name> lora <epoch_N_lora_weights.pt> <gpu>
set -euo pipefail

cd /Data_8TB/lht/PseudoVideo-SAM3-X3-B7
PY=/home/violet/anaconda3/envs/sam3/bin/python
BASE=/Data_8TB/lht/models/modelscope/models/facebook--sam3/snapshots/master/sam3.pt
ROUTES_ROOT=/Data_8TB/lht/PseudoVideo-SAM3-X3-B7/work/kvasir_1pct_anchors/stage1_feature_knn_b7_ft1pct

NAME="$1"
TYPE="$2"
SRC="$3"
GPU="${4:-0}"
export CUDA_VISIBLE_DEVICES="$GPU"

VIDEO_CKPT="work/kvasir_1pct_anchors/video_checkpoints/${NAME}_merged_video.pt"
if [[ ! -f "$VIDEO_CKPT" ]]; then
  echo "[eval] merging $TYPE checkpoint -> $VIDEO_CKPT"
  if [[ "$TYPE" == "lora" ]]; then
    "$PY" scripts/merge_sam3_lora_video_checkpoint.py \
      --base-checkpoint "$BASE" \
      --lora-weights "$SRC" \
      --output "$VIDEO_CKPT"
  else
    "$PY" - "$BASE" "$SRC" "$VIDEO_CKPT" <<'PY'
import json, sys
from pathlib import Path
import torch
base_path, src_path, out_path = sys.argv[1:]
base = torch.load(base_path, map_location="cpu")
ckpt = torch.load(src_path, map_location="cpu", weights_only=False)
detector = ckpt.get("model") if isinstance(ckpt, dict) else None
if detector is None:
    raise RuntimeError(f"{src_path} has no model key")
merged = dict(base)
updated = 0
for key, value in detector.items():
    dst = f"detector.{key}"
    if dst not in merged:
        continue
    if tuple(merged[dst].shape) != tuple(value.shape):
        raise RuntimeError(f"shape mismatch {dst}")
    merged[dst] = value
    updated += 1
Path(out_path).parent.mkdir(parents=True, exist_ok=True)
torch.save(merged, out_path)
print(json.dumps({"updated": updated, "out": out_path}))
PY
  fi
fi

EVAL_ROOT="work/kvasir_1pct_anchors/lora_experiment/test_eval/$NAME"
for mode in anchor_conditioned_target_pooling anchor_conditioned_patch_correspondence; do
  mkdir -p "$EVAL_ROOT/$mode"
  if [[ ! -e "$EVAL_ROOT/$mode/test_pool0_stage1" ]]; then
    ln -s "$ROUTES_ROOT/$mode/test_pool0_stage1" "$EVAL_ROOT/$mode/test_pool0_stage1"
  fi
  echo "[eval] ${NAME} ${mode}"
  "$PY" scripts/stage1_eval_routes_forward_only.py \
    --checkpoint "$VIDEO_CKPT" \
    --mode "$mode" \
    --root "$EVAL_ROOT" \
    --canvas 512 \
    --resume
done

echo "DONE $NAME"
