#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ROOT = Path("/Data_8TB/lht/PseudoVideo-SAM3-X3-B7")
OLD_ROOT = ROOT / "work/kvasir_1pct_anchors/stage1_feature_knn_b7"
OLD_MODES = (
    "t18_corrected",
    "dino_global_pooling",
    "dino_patch_average",
    "anchor_conditioned_target_pooling",
    "anchor_conditioned_patch_correspondence",
)
KEYS = (
    "t18_corrected",
    "dino_global_pooling",
    "dino_patch_average",
    "anchor_conditioned_target_pooling",
    "anchor_conditioned_target_pooling__knn_cls",
    "anchor_conditioned_target_pooling__knn_pooled",
    "anchor_conditioned_target_pooling__knn_cond",
    "anchor_conditioned_patch_correspondence",
    "anchor_conditioned_patch_correspondence__knn_cls",
    "anchor_conditioned_patch_correspondence__knn_pooled",
    "anchor_conditioned_patch_correspondence__knn_cond",
)
BRIDGES = ["direct", "bridge_1", "bridge_2", "bridge_3", "bridge_4", "bridge_5", "bridge_6"]
BRIDGE_LABELS = ["direct", "b1", "b2", "b3", "b4", "b5", "b6"]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def per_bridge_dice(root: Path, key: str, eval_name: str) -> dict[str, float]:
    path = root / key / eval_name / "route_results.jsonl"
    if not path.exists():
        return {b: float("nan") for b in BRIDGES}
    rows = read_jsonl(path)
    buckets: dict[str, list[float]] = {}
    for row in rows:
        buckets.setdefault(row["route_type"], []).append(float(row["gt_dice_evaluation_only"]))
    return {b: float(np.mean(buckets.get(b, []))) if buckets.get(b) else float("nan") for b in BRIDGES}


def short_name(key: str) -> str:
    return key.replace("anchor_conditioned_", "").split("__knn_")[0]


def discover_keys(root: Path) -> tuple[str, ...]:
    return tuple(
        sorted(
            d.name
            for d in root.iterdir()
            if d.is_dir() and (d / "test_pool0_stage1" / "routes.jsonl").exists()
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT / "work/kvasir_1pct_anchors/stage1_feature_knn_vitb256")
    parser.add_argument("--eval-name", type=str, default="eval_base_no_ft_b7_forward")
    parser.add_argument("--label", type=str, default="")
    parser.add_argument("--title", type=str, default="")
    parser.add_argument("--no-old-reference", action="store_true")
    args = parser.parse_args()

    keys = discover_keys(args.root) or KEYS
    rows: list[dict] = []
    for key in keys:
        meta_path = args.root / key / "meta.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        dice = per_bridge_dice(args.root, key, args.eval_name)
        rows.append(
            {
                "mode_key": key,
                "mode": meta.get("mode", key),
                "dinov3_model": meta.get("dinov3_model", ""),
                "feature_size": meta.get("feature_size", 256),
                "knn_feature": meta.get("knn_feature", ""),
                "eval_name": args.eval_name,
                "dice": dice,
                "n": 100,
            }
        )

    # Old reference: vits16 @224 + SAM3 @512 (coarse reference only, not directly comparable)
    old_rows = []
    for mode in OLD_MODES:
        summary_path = OLD_ROOT / mode / "eval_base_no_ft_b7_forward" / "route_family_summary.json"
        if not summary_path.exists():
            continue
        summary = json.loads(summary_path.read_text())
        old_rows.append(
            {
                "mode": mode,
                "dice": {b: summary.get(b, {}).get("dice", float("nan")) for b in BRIDGES},
            }
        )

    args.root.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.label}" if args.label else ""
    protocol = (
        args.title
        or f"Kvasir test, checkpoint={args.eval_name}, forward-only Dice, canvas 256, DINOv3 @256"
    )
    summary = {"protocol": protocol, "rows": rows}
    (args.root / f"comparison_summary{suffix}.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    lines = [
        f"# KNN 实验汇总（{args.label or args.eval_name}）",
        "",
        f"协议：{protocol}，Kvasir test 100 targets，bridge b0–b6。",
        "",
        "| variant | knn | direct | b1 | b2 | b3 | b4 | b5 | b6 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        d = row["dice"]
        label = short_name(row["mode_key"])
        cells = [label, row["knn_feature"]] + [f"{d[b]:.4f}" for b in BRIDGES]
        lines.append("| " + " | ".join(cells) + " |")

    if old_rows and not args.no_old_reference:
        lines += [
            "",
            "## 旧参考（vits16 @224 + SAM3 @512，仅粗参考，分辨率不同不可直接对比）",
            "",
            "| mode | direct | b1 | b2 | b3 | b4 | b5 | b6 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for row in old_rows:
            d = row["dice"]
            cells = [row["mode"]] + [f"{d[b]:.4f}" for b in BRIDGES]
            lines.append("| " + " | ".join(cells) + " |")

    text = "\n".join(lines) + "\n"
    (args.root / f"comparison_summary{suffix}.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
