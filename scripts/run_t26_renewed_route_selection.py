#!/usr/bin/env python3
"""Validation-only selection of a T26 student auditor and SAM3 route score."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

from scipy.stats import spearmanr


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_SCRIPT = PROJECT_ROOT / "scripts/run_t25_offline_analysis.py"
spec = importlib.util.spec_from_file_location("t25_analysis", ANALYSIS_SCRIPT)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot import {ANALYSIS_SCRIPT}")
t25 = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = t25
spec.loader.exec_module(t25)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def linear_candidates(rows: list[dict]) -> list[dict]:
    candidates = []
    for ir in range(21):
        for im in range(21 - ir):
            wr, wm = ir * 0.05, im * 0.05
            ws = round(1.0 - wr - wm, 10)

            def score(row, a=wr, b=wm, c=ws):
                return (
                    a * row["q_return"]
                    + b * row["q_multi"]
                    + c * row["q_model"]
                )

            validation, _ = t25.evaluate(
                rows, f"linear_{wr:.2f}_{wm:.2f}_{ws:.2f}", score
            )
            correlation = float(
                spearmanr(
                    [score(row) for row in rows],
                    [row["gt_dice_evaluation_only"] for row in rows],
                ).statistic
            )
            candidates.append(
                {
                    "weights": {
                        "q_return": wr,
                        "q_multi": wm,
                        "q_model": ws,
                    },
                    "validation": validation,
                    "route_gt_spearman": correlation,
                }
            )
    return candidates


def selection_key(item: dict) -> tuple:
    weights = item["weights"]
    return (
        item["validation"]["dice"],
        item["validation"]["iou"],
        item["route_gt_spearman"],
        -weights["q_return"],
        -weights["q_multi"],
        -weights["q_model"],
        item.get("student", ""),
    )


def score_function(weights: dict):
    def score(row):
        return (
            weights["q_return"] * row["q_return"]
            + weights["q_multi"] * row["q_multi"]
            + weights["q_model"] * row["q_model"]
        )

    return score


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-routes", type=Path, required=True)
    parser.add_argument("--test-routes", type=Path, required=True)
    parser.add_argument(
        "--candidate",
        action="append",
        nargs=3,
        metavar=("LABEL", "VALIDATION_PREDICTIONS", "TEST_PREDICTIONS"),
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    validation_results = []
    candidate_paths = {}
    # Phase 1: choose both checkpoint and score using Validation only.
    for label, validation_predictions, test_predictions in args.candidate:
        candidate_paths[label] = {
            "validation": Path(validation_predictions),
            "test": Path(test_predictions),
        }
        rows = t25.build_rows(
            args.validation_routes, Path(validation_predictions)
        )
        write_jsonl(args.output_dir / label / "q_model_routes_validation.jsonl", rows)
        best = max(linear_candidates(rows), key=selection_key)
        best["student"] = label
        validation_results.append(best)

    winner = max(validation_results, key=selection_key)
    protocol = {
        "selection_rule": (
            "Validation only: Dice, IoU, route Spearman, lower q_return, "
            "deterministic weight/student tie-break"
        ),
        "test_used_for_student_or_weight_selection": False,
        "grid_step": 0.05,
        "grid_size_per_student": 231,
        "candidates": validation_results,
        "winner": winner,
    }
    (args.output_dir / "validation_selection.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    # Phase 2: unlock Test exactly once for the Validation-selected winner.
    winner_label = winner["student"]
    winner_test_rows = t25.build_rows(
        args.test_routes, candidate_paths[winner_label]["test"]
    )
    write_jsonl(
        args.output_dir / winner_label / "q_model_routes_test.jsonl",
        winner_test_rows,
    )
    official_test, official_changes = t25.evaluate(
        winner_test_rows,
        "t26_validation_selected_linear",
        score_function(winner["weights"]),
    )
    write_jsonl(args.output_dir / "official_test_per_query.jsonl", official_changes)
    official = {
        "student": winner_label,
        "weights": winner["weights"],
        "validation": winner["validation"],
        "test": official_test,
        "test_used_for_selection": False,
    }
    (args.output_dir / "official_result.json").write_text(
        json.dumps(official, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    # Phase 3: after the official result is frozen, report sensitivity only.
    sensitivity = {}
    for item in validation_results:
        label = item["student"]
        rows = (
            winner_test_rows
            if label == winner_label
            else t25.build_rows(args.test_routes, candidate_paths[label]["test"])
        )
        linear_test, _ = t25.evaluate(
            rows,
            "validation_selected_linear_sensitivity",
            score_function(item["weights"]),
        )
        b7_test, _ = t25.evaluate(
            rows, "B7_geometric_sensitivity", t25.SELECTORS["B7_geometric"]
        )
        sensitivity[label] = {
            "validation_selected_weights": item["weights"],
            "validation": item["validation"],
            "linear_test": linear_test,
            "b7_test_analysis_only": b7_test,
        }
    (args.output_dir / "sensitivity.json").write_text(
        json.dumps(sensitivity, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"official": official, "sensitivity": sensitivity}))


if __name__ == "__main__":
    main()
