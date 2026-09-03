#!/usr/bin/env python3
"""Freeze validation selection and summarize Round3 T0/T1/T2 results."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Any
import yaml

REPO = Path("/Data_8TB/lht/PseudoVideo-SAM3-X3-B7")

def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO / path

def save(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

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
        "validation_mean_b3_b6": best["mean_b3_b6"], "validation_b6": best["combined"]["b6"],
        "selection_rule": "maximize validation mean(b3,b4,b5,b6); tie-break higher b6; then earlier checkpoint",
        "result_path": best["result_path"], "candidate_count": len(candidates),
    }
    save(run_dir / "best_checkpoint.json", frozen)
    return frozen

def closeout(root: Path) -> dict[str, Any]:
    names = {"T0": "T0_frozen", "T1": "T1_memory", "T2": "T2_full_tracker"}
    results = {}
    for group, name in names.items():
        paths = sorted((root / name / "test").glob("*/b0_b6_test.json"))
        if not paths:
            continue
        results[group] = json.loads(paths[-1].read_text(encoding="utf-8"))
    if "T0" in results:
        baseline = results["T0"]["combined"]
        for group in ("T1", "T2"):
            if group in results:
                results[group]["gain_over_T0"] = {key: results[group]["combined"][key] - baseline[key] for key in baseline}
                results[group]["mean_b3_b6_gain_over_T0"] = results[group]["mean_b3_b6"] - results["T0"]["mean_b3_b6"]
    save(root / "summaries" / "round3_results.json", results)
    with (root / "summaries" / "round3_results.tsv").open("w", encoding="utf-8") as handle:
        handle.write("Model\tb0\tb1\tb2\tb3\tb4\tb5\tb6\tmean_b3-b6\n")
        for group in ("T0", "T1", "T2"):
            if group in results:
                row = results[group]
                handle.write(group + "\t" + "\t".join(f"{row['combined'][f'b{x}']:.9f}" for x in range(7)) + f"\t{row['mean_b3_b6']:.9f}\n")
    return results

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=REPO / "configs/c0_256_round3_tracker_stage4.yaml")
    parser.add_argument("--group", choices=("T1", "T2"))
    parser.add_argument("--closeout", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    root = resolve(config["experiment"]["root"])
    if args.closeout:
        print(json.dumps(closeout(root), indent=2, sort_keys=True))
    else:
        if args.group is None:
            raise RuntimeError("--group is required unless --closeout is used")
        run_dir = root / ({"T1": "T1_memory", "T2": "T2_full_tracker"}[args.group])
        print(json.dumps(select_best(run_dir), indent=2, sort_keys=True))

if __name__ == "__main__":
    main()
