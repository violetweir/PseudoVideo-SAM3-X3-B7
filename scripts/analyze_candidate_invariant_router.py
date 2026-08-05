#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


FEATURE_NAMES = [
    "bridge_count",
    "path_bottleneck_similarity",
    "path_mean_similarity",
    "forward_sam_score",
    "forward_candidate_count",
    "query_attachment_neighbor_rank",
    "anchor_knn_edge",
    "bridge_x_bottleneck",
    "bridge_x_mean",
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def target_id(row: dict[str, Any]) -> str:
    return str(row["target_id"])


def dice_value(row: dict[str, Any]) -> float:
    if "gt_dice_evaluation_only" in row:
        return float(row["gt_dice_evaluation_only"])
    if "_computed_gt_dice_evaluation_only" not in row:
        row["_computed_gt_dice_evaluation_only"] = compute_mask_dice(
            Path(row["forward_mask_path"]),
            Path(row["target_mask_path_evaluation_only"]),
        )
    return float(row["_computed_gt_dice_evaluation_only"])


def compute_mask_dice(pred_path: Path, gt_path: Path) -> float:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "Pillow is required to compute missing gt_dice_evaluation_only from PNG masks"
        ) from exc
    pred = Image.open(pred_path).convert("L")
    gt = Image.open(gt_path).convert("L").resize(pred.size)
    pred_data = [px > 0 for px in pred.getdata()]
    gt_data = [px > 0 for px in gt.getdata()]
    inter = sum(1 for a, b in zip(pred_data, gt_data) if a and b)
    pred_sum = sum(1 for v in pred_data if v)
    gt_sum = sum(1 for v in gt_data if v)
    denom = pred_sum + gt_sum
    return 1.0 if denom == 0 else 2.0 * inter / denom


def route_key(row: dict[str, Any]) -> tuple[str, int, str]:
    return (target_id(row), int(row["bridge_count"]), str(row["route_id"]))


def feature_vector(row: dict[str, Any]) -> list[float]:
    bridge = float(row.get("bridge_count") or 0.0)
    bottleneck = float(row.get("path_bottleneck_similarity") or 0.0)
    mean = float(row.get("path_mean_similarity") or 0.0)
    rank = row.get("query_attachment_neighbor_rank")
    rank_score = 0.0 if rank in (None, "") else 1.0 / max(float(rank), 1.0)
    return [
        bridge,
        bottleneck,
        mean,
        float(row.get("forward_sam_score") or 0.0),
        float(row.get("forward_candidate_count") or 0.0),
        rank_score,
        1.0 if row.get("anchor_knn_edge") else 0.0,
        bridge * bottleneck,
        bridge * mean,
    ]


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


def mean_std(values: list[float]) -> tuple[float, float]:
    mean = sum(values) / max(len(values), 1)
    var = sum((v - mean) ** 2 for v in values) / max(len(values), 1)
    return mean, max(math.sqrt(var), 1e-8)


class RidgeScorer:
    def __init__(self, means: list[float], stds: list[float], weights: list[float]) -> None:
        self.means = means
        self.stds = stds
        self.weights = weights

    @classmethod
    def fit(cls, rows: list[dict[str, Any]], ridge: float) -> "RidgeScorer":
        raw = [feature_vector(row) for row in rows]
        columns = transpose(raw)
        means_stds = [mean_std(col) for col in columns]
        means = [x[0] for x in means_stds]
        stds = [x[1] for x in means_stds]
        x = [[1.0] + [(v - means[i]) / stds[i] for i, v in enumerate(vec)] for vec in raw]
        y = [dice_value(row) for row in rows]
        xt = transpose(x)
        xtx = matmul(xt, x)
        for i in range(1, len(xtx)):
            xtx[i][i] += ridge
        xty = matvec(xt, y)
        return cls(means, stds, solve_linear(xtx, xty))

    def score(self, row: dict[str, Any]) -> float:
        raw = feature_vector(row)
        x = [1.0] + [(v - self.means[i]) / self.stds[i] for i, v in enumerate(raw)]
        return sum(w * v for w, v in zip(self.weights, x))


def spearman(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2:
        return 0.0

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

    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    vx = sum((v - mx) ** 2 for v in rx)
    vy = sum((v - my) ** 2 for v in ry)
    if vx <= 0 or vy <= 0:
        return 0.0
    return sum((a - mx) * (b - my) for a, b in zip(rx, ry)) / math.sqrt(vx * vy)


def group_by_target(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in sorted(rows, key=route_key):
        grouped.setdefault(target_id(row), []).append(row)
    return grouped


def evaluate_pool(
    rows: list[dict[str, Any]],
    scorer: RidgeScorer,
    max_bridge: int,
    dominated_margin: float,
) -> dict[str, Any]:
    selected_dice: list[float] = []
    oracle_dice: list[float] = []
    selected_counts: dict[str, int] = {}
    selected_correct = 0
    dominated_unchanged = 0
    dominated_total = 0
    duplicate_unchanged = 0
    duplicate_total = 0
    all_scores: list[float] = []
    all_dice: list[float] = []

    for target_rows in group_by_target(rows).values():
        candidates = [row for row in target_rows if int(row["bridge_count"]) <= max_bridge]
        if not candidates:
            continue
        scored = [(scorer.score(row), row) for row in candidates]
        selected_score, selected = max(scored, key=lambda item: (item[0], -int(item[1]["bridge_count"]), item[1]["route_id"]))
        oracle = max(candidates, key=lambda row: (dice_value(row), -int(row["bridge_count"]), row["route_id"]))
        selected_dice.append(dice_value(selected))
        oracle_dice.append(dice_value(oracle))
        selected_counts[str(selected["route_type"])] = selected_counts.get(str(selected["route_type"]), 0) + 1
        selected_correct += int(selected["route_id"] == oracle["route_id"])
        all_scores.extend(score for score, _ in scored)
        all_dice.extend(dice_value(row) for _, row in scored)

        dominated = [row for row in candidates if dice_value(row) < dice_value(oracle) - dominated_margin]
        for row in dominated:
            dominated_total += 1
            dominated_unchanged += int(scorer.score(row) <= selected_score or row["route_id"] == selected["route_id"])

        duplicate_total += 1
        duplicated = candidates + [dict(selected, route_id=f"{selected['route_id']}::dup{k}") for k in range(5)]
        dup_selected = max(
            ((scorer.score(row), row) for row in duplicated),
            key=lambda item: (item[0], -int(item[1]["bridge_count"]), item[1]["route_id"]),
        )[1]
        duplicate_unchanged += int(str(dup_selected["route_id"]).split("::dup")[0] == selected["route_id"])

    n = len(selected_dice)
    return {
        "max_bridge": max_bridge,
        "n_targets": n,
        "selected_dice": sum(selected_dice) / max(n, 1),
        "oracle_dice": sum(oracle_dice) / max(n, 1),
        "oracle_gap": (sum(oracle_dice) - sum(selected_dice)) / max(n, 1),
        "selection_accuracy": selected_correct / max(n, 1),
        "spearman_score_dice": spearman(all_scores, all_dice),
        "dominated_candidate_robustness": dominated_unchanged / max(dominated_total, 1),
        "dominated_candidate_count": dominated_total,
        "duplicate_robustness": duplicate_unchanged / max(duplicate_total, 1),
        "selected_route_counts": selected_counts,
    }


def write_table(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "split",
                "max_bridge",
                "n_targets",
                "selected_dice",
                "oracle_dice",
                "oracle_gap",
                "selection_accuracy",
                "spearman_score_dice",
                "dominated_candidate_robustness",
                "duplicate_robustness",
                "selected_route_counts",
            ],
        )
        writer.writeheader()
        for row in rows:
            flat = {key: row.get(key) for key in writer.fieldnames}
            flat["selected_route_counts"] = json.dumps(flat["selected_route_counts"], sort_keys=True)
            writer.writerow(flat)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-results", type=Path, required=True)
    parser.add_argument("--eval-results", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-bridges", type=int, nargs="+", default=[2, 3, 4, 5, 6, 7])
    parser.add_argument("--ridge", type=float, default=1.0)
    parser.add_argument("--dominated-margin", type=float, default=0.10)
    args = parser.parse_args()

    train_rows = read_jsonl(args.train_results)
    eval_rows = read_jsonl(args.eval_results)
    scorer = RidgeScorer.fit(train_rows, args.ridge)
    summaries = []
    for split, rows in (("train", train_rows), ("eval", eval_rows)):
        available_max = max(int(row["bridge_count"]) for row in rows)
        for max_bridge in args.max_bridges:
            if max_bridge <= available_max:
                summary = evaluate_pool(rows, scorer, max_bridge, args.dominated_margin)
                summaries.append({"split": split, **summary})

    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "summary.json").write_text(
        json.dumps(
            {
                "feature_names": FEATURE_NAMES,
                "weights": scorer.weights,
                "means": scorer.means,
                "stds": scorer.stds,
                "summaries": summaries,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    write_table(args.output_root / "summary.csv", summaries)
    print(json.dumps(summaries, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
