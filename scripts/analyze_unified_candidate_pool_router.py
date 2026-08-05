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
    "t18_corrected",
]
DELTAS = [0.0, 0.0025, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def dice(row: dict[str, Any]) -> float:
    return float(row["gt_dice_evaluation_only"])


def route_sequence_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (row["target_id"], row["anchor_id"], tuple(row.get("bridge_ids", [])))


def grouped(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        out.setdefault(row["target_id"], []).append(row)
    return out


def mean(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


def feature_vector(row: dict[str, Any]) -> list[float]:
    bridge = float(row.get("bridge_count") or 0.0)
    bottleneck = float(row.get("path_bottleneck_similarity") or 0.0)
    path_mean = float(row.get("path_mean_similarity") or 0.0)
    rank = row.get("query_attachment_neighbor_rank")
    rank_score = 0.0 if rank in (None, "") else 1.0 / max(float(rank), 1.0)
    mode = row["feature_mode"]
    mode_bits = [1.0 if mode == item else 0.0 for item in MODES]
    return [
        bridge,
        bottleneck,
        path_mean,
        float(row.get("forward_sam_score") or 0.0),
        float(row.get("forward_candidate_count") or 0.0),
        rank_score,
        1.0 if row.get("anchor_knn_edge") else 0.0,
        bridge * bottleneck,
        bridge * path_mean,
        *mode_bits,
        *(bridge * bit for bit in mode_bits),
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


class UnifiedRidge:
    def __init__(self, means: list[float], stds: list[float], weights: list[float]) -> None:
        self.means = means
        self.stds = stds
        self.weights = weights

    @classmethod
    def fit(cls, rows: list[dict[str, Any]], ridge: float) -> "UnifiedRidge":
        raw = [feature_vector(row) for row in rows]
        columns = transpose(raw)
        means = [mean(col) for col in columns]
        stds = [max(math.sqrt(mean([(v - means[i]) ** 2 for v in col])), 1e-8) for i, col in enumerate(columns)]
        x = [[1.0] + [(v - means[i]) / stds[i] for i, v in enumerate(vec)] for vec in raw]
        y = [dice(row) for row in rows]
        xt = transpose(x)
        xtx = matmul(xt, x)
        for i in range(1, len(xtx)):
            xtx[i][i] += ridge
        return cls(means, stds, solve_linear(xtx, matvec(xt, y)))

    def score(self, row: dict[str, Any]) -> float:
        raw = feature_vector(row)
        x = [1.0] + [(v - self.means[i]) / self.stds[i] for i, v in enumerate(raw)]
        return sum(w * v for w, v in zip(self.weights, x))


def load_split(root: Path, split: str, modes: list[str]) -> list[dict[str, Any]]:
    rows = []
    base = root / "work/kvasir_1pct_anchors"
    for mode in modes:
        if split == "validation":
            path = base / "stage1_feature_knn_b7_validation" / mode / "eval_base_no_ft_b7_forward_validation/route_results.jsonl"
        else:
            path = base / "stage1_feature_knn_b7" / mode / "eval_base_no_ft_b7_forward/route_results.jsonl"
        for row in read_jsonl(path):
            enriched = dict(row)
            enriched["feature_mode"] = mode
            rows.append(enriched)
    return rows


def dedupe(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        key = route_sequence_key(row)
        current = by_key.get(key)
        if current is None or row["feature_mode"] < current["feature_mode"]:
            by_key[key] = row
    return list(by_key.values())


def reference_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    refs = {}
    for row in rows:
        if row["feature_mode"] == "anchor_conditioned_target_pooling" and int(row["bridge_count"]) == 6:
            refs[row["target_id"]] = row
    return refs


def evaluate(rows: list[dict[str, Any]], scorer: UnifiedRidge, delta: float | None) -> dict[str, Any]:
    target_rows = {tid: dedupe(items) for tid, items in grouped(rows).items()}
    refs = reference_rows(rows)
    selected = []
    oracle = []
    switched = 0
    hist: dict[str, int] = {}
    for tid, candidates in target_rows.items():
        best = max(candidates, key=lambda row: (scorer.score(row), row["feature_mode"], row["route_id"]))
        if delta is not None:
            ref = refs[tid]
            chosen = best if scorer.score(best) > scorer.score(ref) + delta else ref
            switched += int(chosen is not ref)
        else:
            chosen = best
        selected.append(dice(chosen))
        oracle.append(max(dice(row) for row in candidates))
        key = f"{chosen['feature_mode']}:{chosen['route_type']}"
        hist[key] = hist.get(key, 0) + 1
    return {
        "delta": delta,
        "selected_dice": mean(selected),
        "oracle_dice": mean(oracle),
        "oracle_gap": mean(oracle) - mean(selected),
        "switch_rate": switched / max(len(selected), 1) if delta is not None else 1.0,
        "histogram": hist,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["split", "policy", "delta", "selected_dice", "oracle_dice", "oracle_gap", "switch_rate", "histogram"],
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
    output_root = args.output_root or args.root / "work/kvasir_1pct_anchors/unified_candidate_router_v1"
    output_root.mkdir(parents=True, exist_ok=True)

    validation = load_split(args.root, "validation", MODES)
    test = load_split(args.root, "test", MODES)
    scorer = UnifiedRidge.fit(validation, args.ridge)

    rows = []
    for split, data in [("validation", validation), ("test", test)]:
        free = evaluate(data, scorer, None)
        rows.append({"split": split, "policy": "free_argmax", **free})
        for delta in DELTAS:
            rows.append({"split": split, "policy": "reference_margin", **evaluate(data, scorer, delta)})
    best_validation = max(
        [row for row in rows if row["split"] == "validation" and row["policy"] == "reference_margin"],
        key=lambda row: row["selected_dice"],
    )
    chosen_test = next(
        row
        for row in rows
        if row["split"] == "test"
        and row["policy"] == "reference_margin"
        and row["delta"] == best_validation["delta"]
    )
    payload = {
        "modes": MODES,
        "validation_best_delta": best_validation["delta"],
        "validation_best": best_validation,
        "test_at_validation_best_delta": chosen_test,
        "all_results": rows,
    }
    (output_root / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(output_root / "summary.csv", rows)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
