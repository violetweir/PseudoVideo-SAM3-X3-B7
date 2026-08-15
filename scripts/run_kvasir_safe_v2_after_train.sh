#!/usr/bin/env bash
set -euo pipefail

cd /Data_8TB/lht/PseudoVideo-SAM3-X3-B7

SAM3_PY=/home/violet/anaconda3/envs/sam3/bin/python
BASE_CKPT=/Data_8TB/lht/models/modelscope/models/facebook--sam3/snapshots/master/sam3.pt
TRAIN_DIR=/Data_8TB/lht/DG-GroupUNet/experiments/wacv2027/T11_sam3_lowlabel_ft_kvasir_budgets/finetune_1pct_plus_hq_pseudo_safe_v2_20260806_seed2026
FINAL_CKPT="$TRAIN_DIR/checkpoints/checkpoint_20.pt"
NAME=ft_1pct_plus_hq_pseudo_safe_v2_20260806
VIDEO_CKPT="work/kvasir_1pct_anchors/video_checkpoints/${NAME}_merged_video.pt"
MERGE_JSON="work/kvasir_1pct_anchors/video_checkpoints/${NAME}_merge_summary.json"
MANIFEST=work/kvasir_1pct_anchors/protocol/merged_manifest.jsonl
SUPPORT=work/kvasir_1pct_anchors/protocol/support_manifest.jsonl
ROUTE3_OUT="work/kvasir_1pct_anchors/model_routes/${NAME}"
MAX6_OUT="work/kvasir_1pct_anchors/model_routes_max6/${NAME}"
RESULT_JSON="work/kvasir_1pct_anchors/${NAME}_results.json"
TRAIN_PATTERN=kvasir_1pct_plus_hq_pseudo_safe_v2_20260806_seed2026.yaml

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
mkdir -p "$(dirname "$VIDEO_CKPT")"

echo "[watch] waiting for $FINAL_CKPT"
while [[ ! -f "$FINAL_CKPT" ]]; do
  if ! pgrep -u violet -f "$TRAIN_PATTERN" >/dev/null; then
    echo "[watch] training ended before checkpoint_20.pt; aborting"
    exit 2
  fi
  date
  sleep 120
done

"$SAM3_PY" - "$BASE_CKPT" "$FINAL_CKPT" "$VIDEO_CKPT" "$MERGE_JSON" "$NAME" <<'PY'
import json
import sys
from pathlib import Path

import torch

base_path, src_path, out_path, merge_json, name = sys.argv[1:]
base_path = Path(base_path)
src_path = Path(src_path)
out_path = Path(out_path)
merge_json = Path(merge_json)
base = torch.load(base_path, map_location="cpu")
ckpt = torch.load(src_path, map_location="cpu")
detector = ckpt.get("model") if isinstance(ckpt, dict) else None
if detector is None:
    raise RuntimeError(f"{src_path} has no model key")
merged = dict(base)
updated = 0
skipped = []
for key, value in detector.items():
    dst = f"detector.{key}"
    if dst not in merged:
        skipped.append(key)
        continue
    if tuple(merged[dst].shape) != tuple(value.shape):
        raise RuntimeError(f"shape mismatch {dst}")
    merged[dst] = value
    updated += 1
out_path.parent.mkdir(parents=True, exist_ok=True)
torch.save(merged, out_path)
summary = {
    "name": name,
    "base": str(base_path),
    "source": str(src_path),
    "output": str(out_path),
    "updated_detector_keys": updated,
    "skipped_count": len(skipped),
    "skipped": skipped[:20],
    "epoch": ckpt.get("epoch") if isinstance(ckpt, dict) else None,
    "steps": ckpt.get("steps") if isinstance(ckpt, dict) else None,
}
merge_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps(summary, indent=2, sort_keys=True))
PY

"$SAM3_PY" scripts/run_t21_dynamic_pseudovideo.py \
  --manifest "$MANIFEST" \
  --support-manifest "$SUPPORT" \
  --output-root "$ROUTE3_OUT" \
  --checkpoint "$VIDEO_CKPT" \
  --phase test_pool0 \
  --resume

"$SAM3_PY" scripts/run_kvasir_max6_pseudovideo.py \
  --manifest "$MANIFEST" \
  --support-manifest "$SUPPORT" \
  --output-root "$MAX6_OUT" \
  --checkpoint "$VIDEO_CKPT" \
  --resume

"$SAM3_PY" - "$ROUTE3_OUT/summary.json" "$MAX6_OUT/summary_max5_max6.json" "$RESULT_JSON" "$NAME" <<'PY'
import json
import sys
from pathlib import Path

route3_path, max6_path, out_path, name = sys.argv[1:]
route3_path = Path(route3_path)
max6_path = Path(max6_path)
out_path = Path(out_path)
result = {
    "name": name,
    "route3_summary": json.loads(route3_path.read_text()) if route3_path.exists() else None,
    "max6_summary": json.loads(max6_path.read_text()) if max6_path.exists() else None,
}
out_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
print(json.dumps(result, indent=2, sort_keys=True))
PY

echo "[done] wrote $RESULT_JSON"
