#!/usr/bin/env python3
"""Evaluate a Round3 tracker checkpoint on the frozen Round2A b0-b6 routes."""
from __future__ import annotations
import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any
import numpy as np
import torch
import yaml
from sam3.model_builder import build_sam3_video_model

REPO = Path("/Data_8TB/lht/PseudoVideo-SAM3-X3-B7")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from round3_lora import inject_lora
EVAL_PATH = REPO / "scripts/eval_route_propagation_quality.py"
spec = importlib.util.spec_from_file_location("round3_route_eval", EVAL_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot import {EVAL_PATH}")
route_eval = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = route_eval
spec.loader.exec_module(route_eval)

HISTORICAL = {
    "validation": {"b0": 0.737071, "b1": 0.823352, "b2": 0.849015, "b3": 0.875069, "b4": 0.868348, "b5": 0.882818, "b6": 0.884577},
    "test": {"b0": 0.812347, "b1": 0.825082, "b2": 0.866180, "b3": 0.885061, "b4": 0.894624, "b5": 0.895240, "b6": 0.904065},
}

def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO / path

def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=REPO / "configs/c0_256_round3_tracker_stage4.yaml")
    parser.add_argument("--group", choices=("T0", "T1", "T2", "T3", "T4"), required=True)
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    parser.add_argument("--tracker-checkpoint", type=Path)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--tag", default="")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    base_checkpoint = resolve(config["protocol"]["initialization_checkpoint"])
    run_name = {
        "T0": "T0_frozen", "T1": "T1_memory", "T2": "T2_full_tracker",
        "T3": "T3_memory_attention_lora", "T4": "T4_memory_attention_decoder_lora",
    }[args.group]
    run_dir = resolve(config["experiment"]["root"]) / run_name
    tag = args.tag or (args.tracker_checkpoint.stem if args.tracker_checkpoint else "e33_baseline")
    output_dir = run_dir / args.split / tag
    if args.split == "test" and args.group != "T0":
        best_path = run_dir / "best_checkpoint.json"
        if not best_path.exists():
            raise RuntimeError("Test is blocked until validation best_checkpoint.json exists")
        best = json.loads(best_path.read_text(encoding="utf-8"))
        if args.tracker_checkpoint is None or args.tracker_checkpoint.resolve() != Path(best["checkpoint"]).resolve():
            raise RuntimeError("Test checkpoint is not the frozen validation-selected best checkpoint")
    model = build_sam3_video_model(checkpoint_path=str(base_checkpoint), load_from_HF=False, device="cuda", compile=False)
    checkpoint = None
    if args.tracker_checkpoint:
        checkpoint = torch.load(args.tracker_checkpoint, map_location="cpu", weights_only=False)
        if checkpoint.get("lora") is not None:
            metadata = checkpoint["lora"]
            inject_lora(
                model.tracker, metadata["policy"], int(metadata["rank"]),
                float(metadata["alpha"]), float(metadata["dropout"]),
            )
        model.tracker.load_state_dict(checkpoint["tracker_state"], strict=True)
    model.eval()
    mode_results = {}
    mode_empty_fractions = {}
    for mode in config["evaluation"]["modes"]:
        route_path = resolve(config["protocol"]["route_roots"][mode]) / f"{args.split}_pool0_stage1" / "routes.jsonl"
        routes = sorted(route_eval.read_jsonl(route_path), key=route_eval.route_sort_key)
        mode_output = output_dir / mode
        rows = route_eval.evaluate(model, routes, mode_output, int(config["experiment"]["canvas"]), args.resume, True, True)
        mode_results[mode] = {
            f"b{bridge}": float(np.mean([row["gt_dice_evaluation_only"] for row in rows if int(row["bridge_count"]) == bridge]))
            for bridge in config["evaluation"]["bridges"]
        }
        empty_count = sum(float(row.get("gt_pred_area_ratio_evaluation_only", 0.0)) <= 0.0 for row in rows)
        mode_empty_fractions[mode] = empty_count / max(len(rows), 1)
    combined = {f"b{bridge}": float(np.mean([mode_results[mode][f"b{bridge}"] for mode in mode_results])) for bridge in config["evaluation"]["bridges"]}
    result = {
        "group": args.group, "split": args.split,
        "checkpoint": str(args.tracker_checkpoint.resolve()) if args.tracker_checkpoint else str(base_checkpoint),
        "checkpoint_step": int(checkpoint["global_step"]) if checkpoint is not None else 0,
        "modes": mode_results, "combined": combined,
        "mean_b3_b6": float(np.mean([combined[f"b{x}"] for x in (3, 4, 5, 6)])),
        "long_gain": float(np.mean([combined[f"b{x}"] for x in (3, 4, 5, 6)]) - combined["b0"]),
        "b6_b0_gap": combined["b6"] - combined["b0"],
        "selection_metric": config["training"]["checkpoint_selection"],
        "empty_prediction_fraction_by_mode": mode_empty_fractions,
        "empty_mask_collapse": float(np.mean(list(mode_empty_fractions.values()))) >= 0.95,
    }
    baseline = HISTORICAL[args.split]
    result["delta_over_T0"] = {key: combined[key] - baseline[key] for key in combined}
    result["delta_mean_b3_b6_over_T0"] = result["mean_b3_b6"] - float(np.mean([baseline[f"b{x}"] for x in (3, 4, 5, 6)]))
    if args.group == "T0":
        delta = {key: combined[key] - HISTORICAL[args.split][key] for key in combined}
        result["historical_reference"] = HISTORICAL[args.split]
        result["historical_delta"] = delta
        result["historical_reproduced"] = max(abs(value) for value in delta.values()) <= 5e-6
    save_json(output_dir / f"b0_b6_{args.split}.json", result)
    with (output_dir / f"b0_b6_{args.split}.tsv").open("w", encoding="utf-8") as handle:
        handle.write("model\tb0\tb1\tb2\tb3\tb4\tb5\tb6\tmean_b3_b6\n")
        handle.write(args.group + "\t" + "\t".join(f"{combined[f'b{x}']:.9f}" for x in range(7)) + f"\t{result['mean_b3_b6']:.9f}\n")
    print(json.dumps(result, indent=2, sort_keys=True))

if __name__ == "__main__":
    main()
