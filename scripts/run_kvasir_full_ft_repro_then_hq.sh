#!/usr/bin/env bash
set -euo pipefail

SAM3_ROOT=/Data_8TB/lht/sam3
PV_ROOT=/Data_8TB/lht/PseudoVideo-SAM3-X3-B7
T11_ROOT=/Data_8TB/lht/DG-GroupUNet/experiments/wacv2027/T11_sam3_lowlabel_ft_kvasir_budgets
SAM3_PY=/home/violet/anaconda3/envs/sam3/bin/python
BASE_CKPT=/Data_8TB/lht/models/modelscope/models/facebook--sam3/snapshots/master/sam3.pt
MANIFEST=work/kvasir_1pct_anchors/protocol/merged_manifest.jsonl
SUPPORT=work/kvasir_1pct_anchors/protocol/support_manifest.jsonl
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

mkdir -p "$SAM3_ROOT/sam3/train/configs/kvasir_budgets"
mkdir -p "$PV_ROOT/work/kvasir_1pct_anchors/video_checkpoints"

write_configs() {
  "$SAM3_PY" - <<'PY'
from pathlib import Path

cfg_dir = Path("/Data_8TB/lht/sam3/sam3/train/configs/kvasir_budgets")
orig = (cfg_dir / "kvasir_1pct_ft_savebest.yaml").read_text()
hq = (cfg_dir / "kvasir_1pct_plus_hq_pseudo_ft_savebest.yaml").read_text()

repro = orig.replace(
    "finetune_1pct_seed2026",
    "finetune_1pct_repro_20260806_seed2026",
)
(cfg_dir / "kvasir_1pct_ft_repro_20260806_seed2026.yaml").write_text(repro)

hq_stable = hq.replace(
    "finetune_1pct_plus_hq_pseudo_seed2026",
    "finetune_1pct_plus_hq_pseudo_stable_20260806_seed2026",
)
hq_stable = hq_stable.replace("stable: false", "stable: true", 1)
(cfg_dir / "kvasir_1pct_plus_hq_pseudo_stable_20260806_seed2026.yaml").write_text(hq_stable)

print("wrote configs")
PY
}

train_one() {
  local config_name="$1"
  local train_dir="$2"

  mkdir -p "$train_dir"
  if [[ -f "$train_dir/checkpoints/checkpoint_20.pt" ]]; then
    echo "[train] found final checkpoint, skipping: $train_dir"
    return
  fi

  echo "[train] starting $config_name"
  cd "$SAM3_ROOT"
  "$SAM3_PY" -u sam3/train/train.py \
    -c "configs/kvasir_budgets/$config_name" \
    --use-cluster 0 \
    --num-gpus 1 \
    > "$train_dir/launcher_stdout.log" 2>&1

  if [[ ! -f "$train_dir/checkpoints/checkpoint_20.pt" ]]; then
    echo "[train] missing checkpoint_20.pt after $config_name"
    exit 2
  fi
}

merge_one() {
  local name="$1"
  local src="$2"
  local out="$PV_ROOT/work/kvasir_1pct_anchors/video_checkpoints/${name}_merged_video.pt"
  local merge_json="$PV_ROOT/work/kvasir_1pct_anchors/video_checkpoints/${name}_merge_summary.json"

  echo "[merge] $name"
  "$SAM3_PY" - "$BASE_CKPT" "$src" "$out" "$merge_json" "$name" <<'PY'
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
        raise RuntimeError(
            f"shape mismatch {dst}: {tuple(merged[dst].shape)} vs {tuple(value.shape)}"
        )
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
}

eval_one() {
  local name="$1"
  local ckpt="work/kvasir_1pct_anchors/video_checkpoints/${name}_merged_video.pt"
  local route_out="work/kvasir_1pct_anchors/model_routes/${name}"
  local max6_out="work/kvasir_1pct_anchors/model_routes_max6/${name}"
  local result_json="work/kvasir_1pct_anchors/${name}_results.json"

  cd "$PV_ROOT"
  echo "[eval] 3-route $name"
  "$SAM3_PY" scripts/run_t21_dynamic_pseudovideo.py \
    --manifest "$MANIFEST" \
    --support-manifest "$SUPPORT" \
    --output-root "$route_out" \
    --checkpoint "$ckpt" \
    --phase test_pool0 \
    --resume

  echo "[eval] max6 $name"
  "$SAM3_PY" scripts/run_kvasir_max6_pseudovideo.py \
    --manifest "$MANIFEST" \
    --support-manifest "$SUPPORT" \
    --output-root "$max6_out" \
    --checkpoint "$ckpt" \
    --resume

  "$SAM3_PY" - "$route_out/summary.json" "$max6_out/summary_max5_max6.json" "$result_json" "$name" <<'PY'
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
}

write_configs

REPRO_NAME=ft_1pct_repro_20260806
REPRO_TRAIN="$T11_ROOT/finetune_1pct_repro_20260806_seed2026"
train_one "kvasir_1pct_ft_repro_20260806_seed2026.yaml" "$REPRO_TRAIN"
merge_one "$REPRO_NAME" "$REPRO_TRAIN/checkpoints/checkpoint_20.pt"
eval_one "$REPRO_NAME"

HQ_NAME=ft_1pct_plus_hq_pseudo_stable_20260806
HQ_TRAIN="$T11_ROOT/finetune_1pct_plus_hq_pseudo_stable_20260806_seed2026"
train_one "kvasir_1pct_plus_hq_pseudo_stable_20260806_seed2026.yaml" "$HQ_TRAIN"
merge_one "$HQ_NAME" "$HQ_TRAIN/checkpoints/checkpoint_20.pt"
eval_one "$HQ_NAME"

echo "[done] full reproduction then HQ experiment finished"
