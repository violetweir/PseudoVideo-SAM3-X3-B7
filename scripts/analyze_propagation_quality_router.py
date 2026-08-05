#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


MODES = [
    "anchor_conditioned_target_pooling",
    "anchor_conditioned_patch_correspondence",
]


PROP_FEATURES = [
    "q_cycle",
    "cycle_success",
    "cycle_sam_score",
    "trace_area_min",
    "trace_area_max",
    "trace_area_final",
    "trace_area_max_rel_delta",
    "trace_empty_count",
    "trace_component_max",
    "trace_component_final",
    "trace_centroid_max_step",
    "trace_bbox_w_max_rel_delta",
    "trace_bbox_h_max_rel_delta",
    "trace_adjacent_dice_min",
    "trace_adjacent_dice_mean",
    "trace_adjacent_dice_last",
    "trace_sam_score_min",
    "trace_sam_score_mean",
    "trace_sam_score_final",
    "trace_candidate_count_max",
    "trace_candidate_count_final",
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def dice(row: dict[str, Any]) -> float:
    return float(row["gt_dice_evaluation_only"])


def mean(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


def grouped(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        out.setdefault(row["target_id"], []).append(row)
    return out


def load_quality(root: Path, mode: str, split: str) -> list[dict[str, Any]]:
    if split == "validation":
        path = (
            root
            / "work/kvasir_1pct_anchors/stage1_feature_knn_b7_validation"
            / mode
            / "propagation_quality_validation/propagation_quality.jsonl"
        )
    else:
        path = (
            root
            / "work/kvasir_1pct_anchors/stage1_feature_knn_b7"
            / mode
            / "propagation_quality_test/propagation_quality.jsonl"
        )
    rows = read_jsonl(path)
    for row in rows:
        row["feature_mode"] = mode
    return rows


def feature_vector(row: dict[str, Any], include_mode: bool) -> list[float]:
    bridge = float(row.get("bridge_count") or 0.0)
    bottleneck = float(row.get("path_bottleneck_similarity") or 0.0)
    path_mean = float(row.get("path_mean_similarity") or 0.0)
    base = [
        bridge,
        bottleneck,
        path_mean,
        float(row.get("final_sam_score") or row.get("forward_sam_score") or 0.0),
        float(row.get("final_candidate_count") or row.get("forward_candidate_count") or 0.0),
        bridge * bottleneck,
        bridge * path_mean,
    ]
    prop = [
        1.0 if row.get(name) is True else 0.0 if row.get(name) is False else float(row.get(name) or 0.0)
        for name in PROP_FEATURES
    ]
    if not include_mode:
        return base + prop
    mode_bits = [1.0 if row["feature_mode"] == mode else 0.0 for mode in MODES]
    return base + prop + mode_bits + [bridge * bit for bit in mode_bits]


def transpose(matrix: list[list[float]]) -> list[list[float]]:
    return [list(col) for col in zip(*matrix)]


def matmul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    return [
        [sum(row[k] * b[k][j] for k in range(len(b))) for j in range(len(b[0]))]
        for row in a
    ]


def matvec(a: list[list[float]], x: list[float]) -> list[float]:
    return [sum(row[j] * x[j] for j in range(len(x))) for row in a]


def solve_linear(a: list[list[float]], b: list[float]) -> list[float]:
    n = len(b)
    aug = [a[i][:] + [b[i]] for i in range(n)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot][col]) < 1e-12:
            continue
        aug[col], aug[pivot] = aug[pivot], aug[col]
        scale = aug[col][col]
        aug[col] = [v / scale for v in aug[col]]
        for row in range(n):
            if row == col:
                continue
            factor = aug[row][col]
            if factor:
                aug[row] = [aug[row][j] - factor * aug[col][j] for j in range(n + 1)]
    return [aug[i][-1] for i in range(n)]


class Ridge:
    def __init__(self, means: list[float], stds: list[float], weights: list[float], include_mode: bool) -> None:
        self.means = means
        self.stds = stds
        self.weights = weights
        self.include_mode = include_mode

    @classmethod
    def fit(cls, rows: list[dict[str, Any]], ridge: float, include_mode: bool) -> "Ridge":
        raw = [feature_vector(row, include_mode) for row in rows]
        columns = transpose(raw)
        means = [mean(col) for col in columns]
        stds = [max(math.sqrt(mean([(v - means[i]) ** 2 for v in col])), 1e-8) for i, col in enumerate(columns)]
        x = [[1.0] + [(v - means[i]) / stds[i] for i, v in enumerate(vec)] for vec in raw]
        y = [dice(row) for row in rows]
        xt = transpose(x)
        xtx = matmul(xt, x)
        for i in range(1, len(xtx)):
            xtx[i][i] += ridge
        return cls(means, stds, solve_linear(xtx, matvec(xt, y)), include_mode)

    def score(self, row: dict[str, Any]) -> float:
        raw = feature_vector(row, self.include_mode)
        x = [1.0] + [(v - self.means[i]) / self.stds[i] for i, v in enumerate(raw)]
        return sum(w * v for w, v in zip(self.weights, x))


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
    rx, ry = ranks(xs), ranks(ys)
    mx, my = mean(rx), mean(ry)
    vx = sum((v - mx) ** 2 for v in rx)
    vy = sum((v - my) ** 2 for v in ry)
    if vx <= 0 or vy <= 0:
        return 0.0
    return sum((a - mx) * (b - my) for a, b in zip(rx, ry)) / math.sqrt(vx * vy)


def auroc(scores: list[float], labels: list[int]) -> float:
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return 0.0
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    rank_sum = sum(rank for rank, index in enumerate(order, start=1) if labels[index])
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def evaluate(rows: list[dict[str, Any]], scorer: Ridge) -> dict[str, Any]:
    selected = []
    oracle = []
    hist: dict[str, int] = {}
    correct = 0
    for target_rows in grouped(rows).values():
        chosen = max(target_rows, key=lambda row: (scorer.score(row), -int(row["bridge_count"]), row["route_id"]))
        best = max(target_rows, key=lambda row: dice(row))
        selected.append(dice(chosen))
        oracle.append(dice(best))
        correct += int(chosen["route_id"] == best["route_id"])
        key = f"{chosen['feature_mode']}:{chosen['route_type']}"
        hist[key] = hist.get(key, 0) + 1
    scores = [scorer.score(row) for row in rows]
    dices = [dice(row) for row in rows]
    return {
        "n_targets": len(selected),
        "selected_dice": mean(selected),
        "oracle_dice": mean(oracle),
        "oracle_gap": mean(oracle) - mean(selected),
        "selection_accuracy": correct / max(len(selected), 1),
        "spearman": spearman(scores, dices),
        "failure_auroc": auroc([-score for score in scores], [int(value < 0.2) for value in dices]),
        "histogram": hist,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "experiment",
                "n_targets",
                "selected_dice",
                "oracle_dice",
                "oracle_gap",
                "selection_accuracy",
                "spearman",
                "failure_auroc",
                "histogram",
            ],
        )
        writer.writeheader()
        for row in rows:
            flat = dict(row)
            flat["histogram"] = json.dumps(flat["histogram"], sort_keys=True)
            writer.writerow(flat)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/Data_8TB/lht/PseudoVideo-SAM3-X3-B7"))
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--ridge", type=float, default=1.0)
    args = parser.parse_args()
    output_root = args.output_root or args.root / "work/kvasir_1pct_anchors/propagation_quality_router_v1"
    output_root.mkdir(parents=True, exist_ok=True)

    results = []
    for mode in MODES:
        train = load_quality(args.root, mode, "validation")
        test = load_quality(args.root, mode, "test")
        scorer = Ridge.fit(train, args.ridge, include_mode=False)
        results.append({"experiment": mode, **evaluate(test, scorer)})

    train_union = []
    test_union = []
    for mode in MODES:
        train_union.extend(load_quality(args.root, mode, "validation"))
        test_union.extend(load_quality(args.root, mode, "test"))
    scorer = Ridge.fit(train_union, args.ridge, include_mode=True)
    results.append({"experiment": "target_pooling+patch_correspondence", **evaluate(test_union, scorer)})

    (output_root / "summary.json").write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(output_root / "summary.csv", results)
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
