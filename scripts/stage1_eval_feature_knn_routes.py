#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from sam3.model_builder import build_sam3_video_model


ROOT = Path("/Data_8TB/lht/PseudoVideo-SAM3-X3-B7")
T21_PATH = ROOT / "scripts/run_t21_dynamic_pseudovideo.py"
spec = importlib.util.spec_from_file_location("t21_dynamic", T21_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot import {T21_PATH}")
t21 = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = t21
spec.loader.exec_module(t21)

MODES = (
    "t18_corrected",
    "dino_global_pooling",
    "dino_patch_average",
    "anchor_conditioned_target_pooling",
    "anchor_conditioned_patch_correspondence",
)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def load_mask(path: str, canvas: int) -> np.ndarray:
    return t21.load_mask(path, canvas)


def select_weighted(phase_root: Path, route_results: list[dict], max_bridge: int, canvas: int = 512) -> list[dict]:
    by_target = {}
    for row in route_results:
        if int(row["bridge_count"]) <= max_bridge:
            by_target.setdefault(row["target_id"], []).append(row)
    out = []
    selected_root = phase_root / "selected_masks"
    selected_root.mkdir(parents=True, exist_ok=True)
    for target_id, candidates in sorted(by_target.items()):
        candidates.sort(key=lambda row: int(row["bridge_count"]))
        masks = [load_mask(row["forward_mask_path"], canvas) for row in candidates]
        scored = []
        for idx, row in enumerate(candidates):
            q_multi = float(np.mean([t21.dice(masks[idx], masks[j]) for j in range(len(masks)) if j != idx]))
            quality = 0.5 * float(row["q_return"]) + 0.5 * q_multi
            scored.append((quality, q_multi, row, masks[idx]))
        quality, q_multi, row, mask = max(
            scored,
            key=lambda item: (item[0], item[1], item[2]["q_return"], item[2]["route_id"]),
        )
        gt = load_mask(row["target_mask_path_evaluation_only"], canvas)
        metric = t21.metrics(mask, gt)
        safe_id = target_id.replace("::", "__")
        selected_path = selected_root / f"{safe_id}.png"
        Image.fromarray(mask.astype(np.uint8) * 255).save(selected_path)
        out.append(
            {
                "target_id": target_id,
                "selected_route_type": row["route_type"],
                "selected_bridge_count": row["bridge_count"],
                "q_return": row["q_return"],
                "q_multi": q_multi,
                "quality_score": quality,
                "selected_mask_path": str(selected_path),
                **{f"gt_{key}_evaluation_only": value for key, value in metric.items()},
            }
        )
    write_jsonl(phase_root / "selections.jsonl", out)
    return out


def summarize(rows: list[dict]) -> dict:
    dice = np.asarray([row["gt_dice_evaluation_only"] for row in rows], dtype=np.float64)
    return {
        "n": len(rows),
        "weighted_dice": float(dice.mean()),
        "weighted_iou": float(np.mean([row["gt_iou_evaluation_only"] for row in rows])),
        "route_counts": {
            name: sum(1 for row in rows if row["selected_route_type"] == name)
            for name in ["direct", "bridge_1", "bridge_2", "bridge_3"]
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--root", type=Path, default=ROOT / "work/kvasir_1pct_anchors/stage1_feature_knn")
    args = parser.parse_args()
    mode_root = args.root / args.mode
    routes = read_jsonl(mode_root / "test_pool0_stage1/routes.jsonl")
    eval_root = mode_root / f"eval_{args.model_name}"
    eval_root.mkdir(parents=True, exist_ok=True)
    model = build_sam3_video_model(
        checkpoint_path=str(args.checkpoint.resolve()),
        load_from_HF=False,
        device="cuda",
        compile=False,
    )
    model.eval()
    results = t21.evaluate_routes(model, eval_root, routes, 512, args.resume)
    selections = select_weighted(eval_root, results, 3, 512)
    summary = summarize(selections)
    (eval_root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"mode": args.mode, "model": args.model_name, **summary}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
