#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any


MODES = [
    "t18_corrected",
    "dino_global_pooling",
    "dino_patch_average",
    "anchor_conditioned_target_pooling",
    "anchor_conditioned_patch_correspondence",
]
MAIN_UNION_MODES = [
    "anchor_conditioned_target_pooling",
    "anchor_conditioned_patch_correspondence",
    "t18_corrected",
]


def load_router_module(root: Path) -> Any:
    path = root / "scripts/analyze_candidate_invariant_router.py"
    spec = importlib.util.spec_from_file_location("candidate_router", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def dice(row: dict[str, Any]) -> float:
    return float(row["gt_dice_evaluation_only"])


def grouped(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        out.setdefault(row["target_id"], []).append(row)
    return out


def mean(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


def ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    out = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        rank = (i + j) / 2.0
        for k in range(i, j + 1):
            out[order[k]] = rank
        i = j + 1
    return out


def spearman(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    rx, ry = ranks(xs), ranks(ys)
    mx, my = mean(rx), mean(ry)
    vx = sum((v - mx) ** 2 for v in rx)
    vy = sum((v - my) ** 2 for v in ry)
    if vx <= 0 or vy <= 0:
        return 0.0
    return sum((a - mx) * (b - my) for a, b in zip(rx, ry)) / math.sqrt(vx * vy)


def kendall_tau(xs: list[float], ys: list[float]) -> float:
    concordant = 0
    discordant = 0
    for i in range(len(xs)):
        for j in range(i + 1, len(xs)):
            dx = (xs[i] > xs[j]) - (xs[i] < xs[j])
            dy = (ys[i] > ys[j]) - (ys[i] < ys[j])
            if dx == 0 or dy == 0:
                continue
            if dx == dy:
                concordant += 1
            else:
                discordant += 1
    denom = concordant + discordant
    return 0.0 if denom == 0 else (concordant - discordant) / denom


def auroc(scores: list[float], labels: list[int]) -> float:
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return 0.0
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    rank_sum = 0.0
    for rank, index in enumerate(order, start=1):
        if labels[index]:
            rank_sum += rank
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def select_by_scorer(rows: list[dict[str, Any]], scorer: Any, max_bridge: int) -> dict[str, dict[str, Any]]:
    selections = {}
    for tid, target_rows in grouped(rows).items():
        candidates = [row for row in target_rows if int(row["bridge_count"]) <= max_bridge]
        selections[tid] = max(
            candidates,
            key=lambda row: (
                scorer.score(row),
                -int(row["bridge_count"]),
                row["route_id"],
            ),
        )
    return selections


def mode_summary(mode: str, root: Path, router: Any, failure_threshold: float) -> dict[str, Any]:
    train_path = (
        root
        / "work/kvasir_1pct_anchors/stage1_feature_knn_b7_validation"
        / mode
        / "eval_base_no_ft_b7_forward_validation/route_results.jsonl"
    )
    test_path = (
        root
        / "work/kvasir_1pct_anchors/stage1_feature_knn_b7"
        / mode
        / "eval_base_no_ft_b7_forward/route_results.jsonl"
    )
    train_rows = read_jsonl(train_path)
    test_rows = read_jsonl(test_path)
    scorer = router.RidgeScorer.fit(train_rows, ridge=1.0)

    by_target = grouped(test_rows)
    fixed: dict[int, float] = {}
    for bridge_count in range(8):
        vals = [
            dice(row)
            for row in test_rows
            if int(row["bridge_count"]) == bridge_count
        ]
        fixed[bridge_count] = mean(vals)
    best_bridge, best_fixed = max(fixed.items(), key=lambda item: item[1])

    selected_c3 = select_by_scorer(test_rows, scorer, 3)
    selected_c7 = select_by_scorer(test_rows, scorer, 7)
    selected_dice = mean([dice(row) for row in selected_c7.values()])
    oracle_dice = mean([
        max(dice(row) for row in target_rows)
        for target_rows in by_target.values()
    ])
    hist: dict[str, int] = {}
    for row in selected_c7.values():
        hist[row["route_type"]] = hist.get(row["route_type"], 0) + 1
    change_rate = mean([
        float(selected_c3[tid]["route_id"] != selected_c7[tid]["route_id"])
        for tid in selected_c7
    ])

    scores = [scorer.score(row) for row in test_rows]
    dices = [dice(row) for row in test_rows]
    failure_labels = [int(value < failure_threshold) for value in dices]
    return {
        "mode": mode,
        "best_fixed_route": f"b{best_bridge}" if best_bridge else "direct",
        "best_fixed_dice": best_fixed,
        "selected_c7_dice": selected_dice,
        "oracle_c7_dice": oracle_dice,
        "dynamic_gain": selected_dice - best_fixed,
        "oracle_regret": oracle_dice - selected_dice,
        "selection_histogram": hist,
        "spearman": spearman(scores, dices),
        "kendall": kendall_tau(scores, dices),
        "failure_auroc": auroc([-score for score in scores], failure_labels),
        "failure_rate": mean([float(x) for x in failure_labels]),
        "c3_to_c7_change_rate": change_rate,
    }


def route_sequence_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row["target_id"],
        row["anchor_id"],
        tuple(row.get("bridge_ids", [])),
    )


def union_oracle(root: Path, modes: list[str]) -> dict[str, Any]:
    by_target: dict[str, dict[tuple[Any, ...], dict[str, Any]]] = {}
    for mode in modes:
        rows = read_jsonl(
            root
            / "work/kvasir_1pct_anchors/stage1_feature_knn_b7"
            / mode
            / "eval_base_no_ft_b7_forward/route_results.jsonl"
        )
        for row in rows:
            enriched = dict(row)
            enriched["feature_mode"] = mode
            by_target.setdefault(row["target_id"], {})[route_sequence_key(row)] = enriched

    oracle_rows = []
    candidates_per_target = []
    for tid, candidates in sorted(by_target.items()):
        rows = list(candidates.values())
        candidates_per_target.append(len(rows))
        best = max(rows, key=lambda row: (dice(row), row["feature_mode"], row["route_id"]))
        oracle_rows.append(best)
    hist: dict[str, int] = {}
    for row in oracle_rows:
        key = f"{row['feature_mode']}:{row['route_type']}"
        hist[key] = hist.get(key, 0) + 1
    return {
        "modes": modes,
        "n_targets": len(oracle_rows),
        "mean_unique_candidates": mean(candidates_per_target),
        "min_unique_candidates": min(candidates_per_target),
        "max_unique_candidates": max(candidates_per_target),
        "oracle_dice": mean([dice(row) for row in oracle_rows]),
        "oracle_histogram": hist,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "mode",
        "best_fixed_route",
        "best_fixed_dice",
        "selected_c7_dice",
        "oracle_c7_dice",
        "dynamic_gain",
        "oracle_regret",
        "spearman",
        "kendall",
        "failure_auroc",
        "failure_rate",
        "c3_to_c7_change_rate",
        "selection_histogram",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            flat = {name: row.get(name) for name in fieldnames}
            flat["selection_histogram"] = json.dumps(flat["selection_histogram"], sort_keys=True)
            writer.writerow(flat)


def write_report(path: Path, rows: list[dict[str, Any]], unions: list[dict[str, Any]]) -> None:
    lines = ["# Candidate-Invariant Router v2 Summary", ""]
    lines.append("## Per-Feature Test Metrics")
    lines.append("")
    lines.append("| Feature | Best fixed | Fixed Dice | Selected C7 | Oracle C7 | Dynamic gain | Regret | Spearman | Kendall | Fail AUROC | C3->C7 change |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        lines.append(
            "| {mode} | {best_fixed_route} | {best_fixed_dice:.6f} | {selected_c7_dice:.6f} | "
            "{oracle_c7_dice:.6f} | {dynamic_gain:+.6f} | {oracle_regret:.6f} | "
            "{spearman:.3f} | {kendall:.3f} | {failure_auroc:.3f} | {c3_to_c7_change_rate:.3f} |".format(**row)
        )
    lines.append("")
    lines.append("## Unified Candidate-Pool Oracle")
    lines.append("")
    lines.append("| Modes | Oracle Dice | Mean unique candidates | Range | Oracle histogram |")
    lines.append("|---|---:|---:|---:|---|")
    for row in unions:
        lines.append(
            "| {modes} | {oracle_dice:.6f} | {mean_unique_candidates:.1f} | {min_unique_candidates}-{max_unique_candidates} | `{hist}` |".format(
                modes="+".join(row["modes"]),
                oracle_dice=row["oracle_dice"],
                mean_unique_candidates=row["mean_unique_candidates"],
                min_unique_candidates=row["min_unique_candidates"],
                max_unique_candidates=row["max_unique_candidates"],
                hist=json.dumps(row["oracle_histogram"], sort_keys=True),
            )
        )
    lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/Data_8TB/lht/PseudoVideo-SAM3-X3-B7"))
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--failure-threshold", type=float, default=0.2)
    args = parser.parse_args()
    root = args.root
    output_root = args.output_root or root / "work/kvasir_1pct_anchors/candidate_invariant_router_v2_full_validation_b7_report"
    output_root.mkdir(parents=True, exist_ok=True)
    router = load_router_module(root)

    rows = [mode_summary(mode, root, router, args.failure_threshold) for mode in MODES]
    unions = [
        union_oracle(root, ["anchor_conditioned_target_pooling", "anchor_conditioned_patch_correspondence"]),
        union_oracle(root, MAIN_UNION_MODES),
    ]
    (output_root / "summary.json").write_text(
        json.dumps({"modes": rows, "unified_oracles": unions}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_csv(output_root / "summary.csv", rows)
    write_report(output_root / "report.md", rows, unions)
    print((output_root / "report.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
