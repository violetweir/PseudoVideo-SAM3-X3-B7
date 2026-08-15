#!/usr/bin/env python3
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


ROOT = Path("work/kvasir_1pct_anchors")


PATHS = {
    "train_target_routes": ROOT / "routeco_sam3_v1_routes/anchor_conditioned_target_pooling/train_pool0_stage1/routes.jsonl",
    "train_patch_routes": ROOT / "routeco_sam3_v1_routes/anchor_conditioned_patch_correspondence/train_pool0_stage1/routes.jsonl",
    "train_target_quality": ROOT / "routeco_sam3_v1_routes/anchor_conditioned_target_pooling/propagation_quality_train/propagation_quality.jsonl",
    "train_patch_quality": ROOT / "routeco_sam3_v1_routes/anchor_conditioned_patch_correspondence/propagation_quality_train/propagation_quality.jsonl",
    "manifest": ROOT / "routeco_sam3_v1/routeco_v1_pseudo_manifest.jsonl",
    "val_routeco_target": ROOT / "stage1_feature_knn_b7_validation/anchor_conditioned_target_pooling/eval_routeco_core_v1_step400_validation/route_results.jsonl",
    "val_routeco_patch": ROOT / "stage1_feature_knn_b7_validation/anchor_conditioned_patch_correspondence/eval_routeco_core_v1_step400_validation/route_results.jsonl",
    "val_step0_target": ROOT / "stage1_feature_knn_b7_validation/anchor_conditioned_target_pooling/eval_routeco_core_v1_step0_validation/route_results.jsonl",
    "val_step0_patch": ROOT / "stage1_feature_knn_b7_validation/anchor_conditioned_patch_correspondence/eval_routeco_core_v1_step0_validation/route_results.jsonl",
    "val_frozen_target": ROOT / "stage1_feature_knn_b7_validation/anchor_conditioned_target_pooling/eval_base_no_ft_b7_forward_validation/route_results.jsonl",
    "val_frozen_patch": ROOT / "stage1_feature_knn_b7_validation/anchor_conditioned_patch_correspondence/eval_base_no_ft_b7_forward_validation/route_results.jsonl",
}


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def stat_rows(rows: list[dict]) -> dict:
    return {
        "rows": len(rows),
        "targets": len({row.get("target_id") for row in rows}),
        "bridge_counts": dict(sorted(Counter(str(row.get("bridge_count")) for row in rows).items())),
        "status": dict(Counter(row.get("status", "NA") for row in rows)),
    }


def dice_stats(rows: list[dict]) -> dict:
    if not rows:
        return {}
    values = [float(row.get("gt_dice_evaluation_only", 0.0)) for row in rows]
    return {
        "zero_routes": sum(value == 0.0 for value in values),
        "nonzero_routes": sum(value > 0.0 for value in values),
        "mean_all_routes": sum(values) / len(values),
    }


def oracle(rows: list[dict]) -> float:
    by_target: dict[str, list[dict]] = {}
    for row in rows:
        if 3 <= int(row["bridge_count"]) <= 6:
            by_target.setdefault(row["target_id"], []).append(row)
    best = [max(items, key=lambda row: float(row["gt_dice_evaluation_only"])) for items in by_target.values()]
    return sum(float(row["gt_dice_evaluation_only"]) for row in best) / max(len(best), 1)


def main() -> None:
    rows = {name: read_jsonl(path) for name, path in PATHS.items()}
    report = {
        name: {**stat_rows(value), "path": str(PATHS[name])}
        for name, value in rows.items()
    }
    manifest = rows["manifest"]
    report["manifest_selected_counts"] = dict(
        Counter(f"{row.get('feature_mode')}:{row.get('route_type')}" for row in manifest)
    )
    route_sets = {
        name: {row.get("route_id") for row in value}
        for name, value in rows.items()
        if name.startswith("val_")
    }
    target_sets = {
        name: {row.get("target_id") for row in value}
        for name, value in rows.items()
        if name.startswith("val_")
    }
    report["validation_intersections"] = {
        "routeco_target_vs_step0_target_routes": len(route_sets["val_routeco_target"] & route_sets["val_step0_target"]),
        "routeco_patch_vs_step0_patch_routes": len(route_sets["val_routeco_patch"] & route_sets["val_step0_patch"]),
        "routeco_target_vs_frozen_target_routes": len(route_sets["val_routeco_target"] & route_sets["val_frozen_target"]),
        "routeco_patch_vs_frozen_patch_routes": len(route_sets["val_routeco_patch"] & route_sets["val_frozen_patch"]),
        "routeco_target_vs_frozen_target_targets": len(target_sets["val_routeco_target"] & target_sets["val_frozen_target"]),
        "routeco_patch_vs_frozen_patch_targets": len(target_sets["val_routeco_patch"] & target_sets["val_frozen_patch"]),
    }
    report["dice_stats"] = {
        name: dice_stats(value)
        for name, value in rows.items()
        if name.startswith("val_")
    }
    report["oracles"] = {
        "routeco_core_step400_target": oracle(rows["val_routeco_target"]),
        "routeco_core_step400_patch": oracle(rows["val_routeco_patch"]),
        "routeco_core_step0_target": oracle(rows["val_step0_target"]),
        "routeco_core_step0_patch": oracle(rows["val_step0_patch"]),
        "frozen_official_target_b3_b6": oracle(rows["val_frozen_target"]),
        "frozen_official_patch_b3_b6": oracle(rows["val_frozen_patch"]),
    }
    output = ROOT / "routeco_sam3_v1/fairness_accounting_routeco_v1.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
