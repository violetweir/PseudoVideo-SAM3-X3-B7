#!/usr/bin/env bash
# Round-2 SAM3 fine-tune evaluation for BOTH final and best-validation
# checkpoints: merge -> propagation quality (validation+test) -> b3-b6 router.
set -u

cd /Data_8TB/lht/PseudoVideo-SAM3-X3-B7
PY=/home/violet/anaconda3/envs/sam3/bin/python
EXP=/Data_8TB/lht/DG-GroupUNet/experiments/wacv2027/T11_sam3_lowlabel_ft_kvasir_budgets/finetune_1pct_round2_student_audited_20260807_seed2026
BASE=/Data_8TB/lht/models/modelscope/models/facebook--sam3/snapshots/master/sam3.pt
OUT=work/kvasir_1pct_anchors/round2
LOG=$OUT/log
mkdir -p "$LOG" "$OUT"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG/eval.log"; }

merge_video() {
  local src="$1" out="$2"
  "$PY" - "$BASE" "$src" "$out" <<'PY'
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
}

eval_variant() {
  local name="$1" ckpt="$2" root="$3"
  local merged="work/kvasir_1pct_anchors/video_checkpoints/${name}_merged_video.pt"
  log "== variant ${name} =="
  if [ ! -f "$merged" ]; then
    log "merge ${name}"
    merge_video "$ckpt" "$merged"
  fi
  for mode in anchor_conditioned_target_pooling anchor_conditioned_patch_correspondence; do
    for split in validation test; do
      if [ ! -f "$root/$mode/propagation_quality_${split}/propagation_quality.jsonl" ]; then
        log "pq ${name} ${mode} ${split}"
        "$PY" scripts/eval_route_propagation_quality.py \
          --checkpoint "$merged" --mode "$mode" --root "$root" \
          --split "$split" --canvas 512 --resume >> "$LOG/pq_${name}.log" 2>&1
      fi
    done
  done
  log "router report ${name}"
  "$PY" scripts/eval_ft1pct_pq_router.py \
    --root "$root" --min-bridge 3 --max-bridge 6 \
    --output "$OUT/router_b3_b6_${name}_report.json" | tee "$OUT/router_b3_b6_${name}_stdout.txt"
}

FINAL_CKPT="$EXP/checkpoints/checkpoint_20.pt"
BEST_CKPT="$EXP/checkpoints/val_roboflow100_detection_coco_eval_segm_AP.pt"

log "wait training complete (checkpoint_20.pt + process exit)"
while [ ! -f "$FINAL_CKPT" ]; do
  log "waiting final ckpt: $(ls "$EXP/checkpoints/" 2>/dev/null | tr '\n' ' ')"
  sleep 600
done
while pgrep -f "kvasir_1pct_round2_student_audited_20260807_seed2026.yaml" >/dev/null; do
  sleep 120
done
log "training done; best ckpt present: $([ -f "$BEST_CKPT" ] && echo yes || echo no)"
if [ ! -f "$BEST_CKPT" ]; then
  log "ERROR: best-val checkpoint missing"
  exit 2
fi

eval_variant final "$FINAL_CKPT" work/kvasir_1pct_anchors/stage1_feature_knn_b7_ftround2
eval_variant bestval "$BEST_CKPT" work/kvasir_1pct_anchors/stage1_feature_knn_b7_ftround2_bestval

log "ROUND2_EVAL_DONE"
