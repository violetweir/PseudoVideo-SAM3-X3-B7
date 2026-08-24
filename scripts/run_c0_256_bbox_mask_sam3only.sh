#!/usr/bin/env bash
set -euo pipefail

cd /Data_8TB/lht/PseudoVideo-SAM3-X3-B7

export PYTHONPATH=/Data_8TB/lht/sam3
PY=/home/violet/anaconda3/envs/sam3/bin/python
ROOT=work/rerun_c0_256/stage1_feature_knn_b7_ft1pct
OUT=work/rerun_c0_256_bbox_mask_sam3only
CKPT=work/kvasir_1pct_anchors/video_checkpoints/ft_1pct_merged_video.pt

mkdir -p "$OUT/logs"

for MODE in anchor_conditioned_target_pooling anchor_conditioned_patch_correspondence; do
  "$PY" scripts/eval_route_propagation_quality_det_gtmask.py \
    --checkpoint "$CKPT" \
    --mode "$MODE" \
    --root "$ROOT" \
    --output-root "$OUT" \
    --split test \
    --canvas 256 \
    --resume \
    > "$OUT/logs/${MODE}.log" 2>&1
done

"$PY" - <<'PY' > "$OUT/logs/compare_bbox_vs_bbox_mask.log"
import json
from pathlib import Path

root = Path("work/rerun_c0_256/stage1_feature_knn_b7_ft1pct")
out = Path("work/rerun_c0_256_bbox_mask_sam3only")
modes = [
    "anchor_conditioned_target_pooling",
    "anchor_conditioned_patch_correspondence",
]

report = {}
for mode in modes:
    bbox = json.loads((root / mode / "propagation_quality_test" / "summary.json").read_text())
    mask = json.loads((out / mode / "propagation_quality_test" / "summary.json").read_text())
    rows = []
    for bridge in range(7):
        key = f"bridge_{bridge}"
        b = bbox[key]
        m = mask[key]
        rows.append({
            "bridge": bridge,
            "bbox_n": b["n"],
            "bbox_dice": b["dice"],
            "bbox_q_cycle": b["q_cycle"],
            "bbox_mask_n": m["n"],
            "bbox_mask_dice": m["dice"],
            "bbox_mask_q_cycle": m["q_cycle"],
            "delta_dice": m["dice"] - b["dice"],
            "delta_q_cycle": m["q_cycle"] - b["q_cycle"],
        })
    report[mode] = rows

(out / "bbox_vs_bbox_mask_summary.json").write_text(
    json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
)

for mode, rows in report.items():
    print(mode)
    print("bridge n bbox bbox+mask delta q_bbox q_bbox+mask delta_q")
    for r in rows:
        print(
            f"b{r['bridge']} {r['bbox_mask_n']} "
            f"{r['bbox_dice']:.6f} {r['bbox_mask_dice']:.6f} {r['delta_dice']:+.6f} "
            f"{r['bbox_q_cycle']:.6f} {r['bbox_mask_q_cycle']:.6f} {r['delta_q_cycle']:+.6f}"
        )
    print()
PY
