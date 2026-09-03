#!/usr/bin/env python3
"""Fixed 20-target normal-vs-shuffled-bridge path sensitivity diagnostic."""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import random
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
spec = importlib.util.spec_from_file_location("round3_path_route_eval", EVAL_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot import {EVAL_PATH}")
route_eval = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = route_eval
spec.loader.exec_module(route_eval)

RUN_NAMES = {"T3": "T3_memory_attention_lora", "T4": "T4_memory_attention_decoder_lora"}


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO / path


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def shuffled_route(route: dict[str, Any], seed: int) -> dict[str, Any]:
    result = copy.deepcopy(route)
    pairs = list(zip(result["bridge_ids"], result["bridge_image_paths"]))
    rng = random.Random(f"{seed}:{route['route_id']}")
    rng.shuffle(pairs)
    if len(pairs) > 1 and pairs == list(zip(route["bridge_ids"], route["bridge_image_paths"])):
        pairs = pairs[1:] + pairs[:1]
    result["bridge_ids"] = [item[0] for item in pairs]
    result["bridge_image_paths"] = [item[1] for item in pairs]
    result["route_id"] = f"shuffle_{route['route_id']}"
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--group", choices=tuple(RUN_NAMES), required=True)
    parser.add_argument("--tracker-checkpoint", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    path_cfg = config["path_sensitivity"]
    target_count = int(path_cfg["target_count"])
    seed = int(path_cfg["seed"])
    checkpoint = torch.load(args.tracker_checkpoint, map_location="cpu", weights_only=False)
    model = build_sam3_video_model(
        checkpoint_path=str(resolve(config["protocol"]["initialization_checkpoint"])),
        load_from_HF=False, device="cuda", compile=False,
    )
    metadata = checkpoint.get("lora")
    if metadata is None:
        raise RuntimeError("Path sensitivity requires a LoRA checkpoint")
    inject_lora(model.tracker, metadata["policy"], int(metadata["rank"]), float(metadata["alpha"]), float(metadata["dropout"]))
    model.tracker.load_state_dict(checkpoint["tracker_state"], strict=True)
    model.eval()

    routes_by_mode = {}
    common_targets = None
    for mode in config["evaluation"]["modes"]:
        route_path = resolve(config["protocol"]["route_roots"][mode]) / "validation_pool0_stage1/routes.jsonl"
        routes = [row for row in route_eval.read_jsonl(route_path) if int(row["bridge_count"]) == 6]
        by_target = {row["target_id"]: row for row in routes}
        routes_by_mode[mode] = by_target
        targets = set(by_target)
        common_targets = targets if common_targets is None else common_targets & targets
    candidates = sorted(common_targets or [])
    rng = random.Random(seed)
    selected_targets = sorted(rng.sample(candidates, target_count))
    root = resolve(config["experiment"]["root"]) / RUN_NAMES[args.group] / "path_sensitivity" / args.tag
    mode_metrics = {}
    for mode, by_target in routes_by_mode.items():
        normal_routes = [by_target[target] for target in selected_targets]
        shuffle_routes = [shuffled_route(row, seed) for row in normal_routes]
        normal_rows = route_eval.evaluate(model, normal_routes, root / mode / "normal", int(config["experiment"]["canvas"]), args.resume, True, True)
        shuffle_rows = route_eval.evaluate(model, shuffle_routes, root / mode / "shuffle", int(config["experiment"]["canvas"]), args.resume, True, True)
        normal = float(np.mean([row["gt_dice_evaluation_only"] for row in normal_rows]))
        shuffle = float(np.mean([row["gt_dice_evaluation_only"] for row in shuffle_rows]))
        mode_metrics[mode] = {"normal": normal, "shuffle": shuffle, "normal_minus_shuffle": normal - shuffle}
    normal = float(np.mean([row["normal"] for row in mode_metrics.values()]))
    shuffle = float(np.mean([row["shuffle"] for row in mode_metrics.values()]))
    result = {
        "group": args.group, "checkpoint": str(args.tracker_checkpoint.resolve()),
        "checkpoint_step": int(checkpoint["global_step"]), "bridge_count": 6,
        "target_count": target_count, "target_ids": selected_targets,
        "shuffle_definition": "Deterministic non-identity permutation of the six correct bridge frames within each route.",
        "modes": mode_metrics, "normal": normal, "shuffle": shuffle,
        "normal_minus_shuffle": normal - shuffle,
    }
    save_json(root / "path_sensitivity.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

