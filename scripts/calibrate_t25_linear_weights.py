#!/usr/bin/env python3
"""Select T25 linear weights on Validation and apply them once to Test."""

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--student", required=True)
    parser.add_argument("--validation-routes", type=Path, required=True)
    parser.add_argument("--validation-predictions", type=Path, required=True)
    parser.add_argument("--test-q-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    validation = t25.build_rows(args.validation_routes, args.validation_predictions)
    test = t25.read_jsonl(args.test_q_model)
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

            validation_result, _ = t25.evaluate(
                validation, f"linear_{wr:.2f}_{wm:.2f}_{ws:.2f}", score
            )
            route_scores = [score(row) for row in validation]
            route_gt = [row["gt_dice_evaluation_only"] for row in validation]
            rank_correlation = float(spearmanr(route_scores, route_gt).statistic)
            candidates.append(
                {
                    "weights": {
                        "q_return": wr,
                        "q_multi": wm,
                        "q_model": ws,
                    },
                    "validation": validation_result,
                    "route_gt_spearman": rank_correlation,
                }
            )
    # Protocol tie order: Validation Dice, route rank correlation, smaller wr,
    # then a deterministic lexicographic order.
    best = max(
        candidates,
        key=lambda item: (
            item["validation"]["dice"],
            item["route_gt_spearman"],
            -item["weights"]["q_return"],
            -item["weights"]["q_multi"],
            -item["weights"]["q_model"],
        ),
    )
    weights = best["weights"]

    def frozen_score(row):
        return (
            weights["q_return"] * row["q_return"]
            + weights["q_multi"] * row["q_multi"]
            + weights["q_model"] * row["q_model"]
        )

    test_result, changes = t25.evaluate(
        test, "validation_calibrated_linear", frozen_score
    )
    payload = {
        "student": args.student,
        "selection_rule": "Validation only; grid step 0.05",
        "test_used_for_weight_selection": False,
        "weights": weights,
        "validation": best["validation"],
        "validation_route_gt_spearman": best["route_gt_spearman"],
        "test": test_result,
        "grid_size": len(candidates),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    changes_path = args.output.with_name(args.output.stem + "_per_query.jsonl")
    with changes_path.open("w") as handle:
        for row in changes:
            handle.write(json.dumps(row) + "\n")
    print(json.dumps(payload))


if __name__ == "__main__":
    main()
