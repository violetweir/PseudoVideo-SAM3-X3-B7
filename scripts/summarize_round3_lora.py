#!/usr/bin/env python3
"""Select validation-best T3/T4 checkpoints and close out the LoRA experiment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

REPO = Path("/Data_8TB/lht/PseudoVideo-SAM3-X3-B7")
RUN_NAMES = {"T3": "T3_memory_attention_lora", "T4": "T4_memory_attention_decoder_lora"}


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO / path


def save(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def select_best(run_dir: Path) -> dict[str, Any]:
    candidates = []
    for path in sorted((run_dir / "validation").glob("*/b0_b6_validation.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        row["result_path"] = str(path)
        candidates.append(row)
    if not candidates:
        raise RuntimeError(f"No validation results under {run_dir}")
    candidates.sort(key=lambda row: (-row["mean_b3_b6"], -row["combined"]["b6"], row["checkpoint_step"]))
    best = candidates[0]
    frozen = {
        "checkpoint": best["checkpoint"], "step": best["checkpoint_step"],
        "validation_mean_b3_b6": best["mean_b3_b6"],
        "validation_b6": best["combined"]["b6"],
        "validation_b6_b0": best["b6_b0_gap"],
        "delta_mean_b3_b6_over_T0": best["delta_mean_b3_b6_over_T0"],
        "empty_mask_collapse": best["empty_mask_collapse"],
        "selection_rule": "maximize validation mean(b3,b4,b5,b6); tie-break higher b6; then earlier checkpoint",
        "result_path": best["result_path"], "candidate_count": len(candidates),
    }
    save(run_dir / "best_checkpoint.json", frozen)
    return frozen


def closeout(root: Path) -> dict[str, Any]:
    baseline_path = REPO / "work/rerun_c0_256_round3_tracker_stage4/T0_frozen/test/e33_baseline/b0_b6_test.json"
    results = {"T0": json.loads(baseline_path.read_text(encoding="utf-8"))}
    for group, name in RUN_NAMES.items():
        path = root / name / "test/best/b0_b6_test.json"
        if path.exists():
            results[group] = json.loads(path.read_text(encoding="utf-8"))
    save(root / "summaries/round3_lora_results.json", results)
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--group", choices=tuple(RUN_NAMES))
    parser.add_argument("--closeout", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    root = resolve(config["experiment"]["root"])
    if args.closeout:
        result = closeout(root)
    else:
        if args.group is None:
            raise RuntimeError("--group is required unless --closeout is used")
        result = select_best(root / RUN_NAMES[args.group])
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

