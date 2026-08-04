#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def route_sort_key(row: dict[str, Any]) -> tuple[str, int, str]:
    return (row["target_id"], int(row["bridge_count"]), row["route_id"])


def evaluate_forward_only(
    model: Any,
    eval_root: Path,
    routes: list[dict[str, Any]],
    canvas: int,
    resume: bool,
) -> list[dict[str, Any]]:
    result_path = eval_root / "route_results.jsonl"
    old_rows = read_jsonl(result_path) if resume else []
    existing = {
        row["route_id"]: row
        for row in old_rows
        if row.get("status") == "success"
        and Path(row.get("forward_mask_path", "")).exists()
        and "gt_dice_evaluation_only" in row
    }
    pending = [row for row in routes if row["route_id"] not in existing]
    forward_root = eval_root / "forward_masks"
    forward_root.mkdir(parents=True, exist_ok=True)
    for position, route in enumerate(pending, 1):
        started = time.time()
        forward_paths = [
            route["anchor_image_path"],
            *route["bridge_image_paths"],
            route["target_image_path"],
        ]
        forward = t21.propagate(
            model,
            forward_paths,
            route["anchor_box_xywh_normalized"],
            canvas,
        )
        forward_path = forward_root / f"{route['route_id']}.png"
        Image.fromarray(forward["mask"].astype(np.uint8) * 255).save(forward_path)
        gt = t21.load_mask(route["target_mask_path_evaluation_only"], canvas)
        metric = t21.metrics(forward["mask"], gt)
        result = {
            **route,
            "status": "success",
            "forward_candidate_count": forward["candidate_count"],
            "forward_sam_score": forward["sam_score"],
            "forward_mask_path": str(forward_path),
            "forward_mask_sha256": t21.sha256_file(forward_path),
            "seconds": round(time.time() - started, 3),
            **{f"gt_{key}_evaluation_only": value for key, value in metric.items()},
        }
        t21.append_fsync(result_path, result)
        existing[route["route_id"]] = result
        print(
            f"[{position}/{len(pending)}] {route['target_id']} "
            f"{route['route_type']} dice={metric['dice']:.4f}",
            flush=True,
        )
    final_rows = [existing[row["route_id"]] for row in routes]
    write_jsonl(result_path, final_rows)
    return final_rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_route: dict[str, list[float]] = {}
    for row in rows:
        by_route.setdefault(row["route_type"], []).append(float(row["gt_dice_evaluation_only"]))
    return {
        route: {
            "n": len(values),
            "dice": float(np.mean(values)),
        }
        for route, values in sorted(by_route.items(), key=lambda item: (item[0] != "direct", item[0]))
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--canvas", type=int, default=512)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    mode_root = args.root / args.mode
    routes = sorted(read_jsonl(mode_root / "test_pool0_stage1/routes.jsonl"), key=route_sort_key)
    eval_root = mode_root / "eval_base_no_ft_b7_forward"
    eval_root.mkdir(parents=True, exist_ok=True)
    model = build_sam3_video_model(
        checkpoint_path=str(args.checkpoint.resolve()),
        load_from_HF=False,
        device="cuda",
        compile=False,
    )
    model.eval()
    rows = evaluate_forward_only(model, eval_root, routes, args.canvas, args.resume)
    summary = summarize(rows)
    (eval_root / "route_family_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"mode": args.mode, "summary": summary}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
