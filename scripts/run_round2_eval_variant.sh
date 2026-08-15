#!/usr/bin/env bash
# Evaluate ONE round-2 checkpoint variant (merge -> propagation quality -> router).
# Usage: run_round2_eval_variant.sh <name> <ckpt> <root>
set -eu

cd /Data_8TB/lht/PseudoVideo-SAM3-X3-B7
PY=/home/violet/anaconda3/envs/sam3/bin/python
BASE=/Data_8TB/lht/models/modelscope/models/facebook--sam3/snapshots/master/sam3.pt
OUT=work/kvasir_1pct_anchors/round2
LOG=$OUT/log
NAME="$1"; CKPT="$2"; ROOT="$3"
MIN_BRIDGE="${MIN_BRIDGE:-3}"
MAX_BRIDGE="${MAX_BRIDGE:-6}"
mkdir -p "$LOG" "$OUT"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG/eval_${NAME}.log"; }

merged="work/kvasir_1pct_anchors/video_checkpoints/${NAME}_merged_video.pt"
log "== variant ${NAME} =="
if [ ! -f "$merged" ]; then
  log "merge"
  "$PY" - "$BASE" "$CKPT" "$merged" <<'PY'
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

for mode in anchor_conditioned_target_pooling anchor_conditioned_patch_correspondence; do
  for split in validation test; do
    if [ ! -f "$ROOT/$mode/propagation_quality_${split}/propagation_quality.jsonl" ]; then
      log "pq ${mode} ${split}"
      "$PY" scripts/eval_route_propagation_quality.py \
        --checkpoint "$merged" --mode "$mode" --root "$ROOT" \
        --split "$split" --canvas 512 --resume >> "$LOG/pq_${NAME}.log" 2>&1
    fi
  done
done

log "router report"
"$PY" scripts/eval_ft1pct_pq_router.py \
  --root "$ROOT" --min-bridge "$MIN_BRIDGE" --max-bridge "$MAX_BRIDGE" \
  --output "$OUT/router_b${MIN_BRIDGE}_b${MAX_BRIDGE}_${NAME}_report.json" | tee "$OUT/router_b${MIN_BRIDGE}_b${MAX_BRIDGE}_${NAME}_stdout.txt"

log "VARIANT_DONE ${NAME}"
