#!/usr/bin/env python3
"""Route-selector methods (M1 fixed 0.5/0.5, M2 calibrated linear, M3 ridge
router, M4 single-signal) on the ViT-B @256 route root.

Goal: maximize test selected Dice.  Validation is used only to calibrate/fit;
test is reported once.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path("/Data_8TB/lht/PseudoVideo-SAM3-X3-B7")
PQR = ROOT / "scripts/analyze_propagation_quality_router.py"
spec = importlib.util.spec_from_file_location("pqr", PQR)
pqr = importlib.util.module_from_spec(spec)
sys.modules["pqr"] = pqr
spec.loader.exec_module(pqr)

KEYS = (
    "t18_corrected",
    "dino_global_pooling",
    "dino_patch_average",
    "anchor_conditioned_target_pooling",
    "anchor_conditioned_target_pooling__knn_cls",
    "anchor_conditioned_target_pooling__knn_pooled",
    "anchor_conditioned_target_pooling__knn_cond",
    "anchor_conditioned_patch_correspondence",
    "anchor_conditioned_patch_correspondence__knn_cls",
    "anchor_conditioned_patch_correspondence__knn_pooled",
    "anchor_conditioned_patch_correspondence__knn_cond",
)
pqr.MODES = list(KEYS)

BRIDGES = [0, 1, 2, 3, 4, 5, 6]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_mask(path: Path, size: int = 256) -> np.ndarray:
    image = Image.open(path).convert("L")
    if image.size != (size, size):
        image = image.resize((size, size), Image.Resampling.NEAREST)
    return np.asarray(image) > 127


def dice(a: np.ndarray, b: np.ndarray) -> float:
    inter = int(np.logical_and(a, b).sum())
    return float(2 * inter / max(int(a.sum()) + int(b.sum()), 1))


def load_quality(root: Path, key: str, split: str, tag: str) -> list[dict]:
    path = root / key / f"propagation_quality_{split}_{tag}" / "propagation_quality.jsonl"
    rows = read_jsonl(path)
    for row in rows:
        row["feature_mode"] = key
        row["family"] = key
    return rows


def pooled(rows: list[dict], bridges: list[int]) -> list[dict]:
    return [r for r in rows if int(r["bridge_count"]) in bridges]


def q_multi_map(rows: list[dict]) -> dict[tuple, float]:
    by_target: dict[str, list[dict]] = {}
    for r in rows:
        by_target.setdefault(r["target_id"], []).append(r)
    out: dict[tuple, float] = {}
    for target, cands in by_target.items():
        masks = {}
        for r in cands:
            p = Path(r["forward_mask_path"])
            if not p.is_absolute():
                p = ROOT / p
            masks[(r["family"], r["route_id"])] = load_mask(p)
        for r in cands:
            key = (r["family"], r["route_id"])
            others = [o for o in cands if (o["family"], o["route_id"]) != key]
            out[key] = float(np.mean([dice(masks[key], masks[(o["family"], o["route_id"])]) for o in others]))
    return out


def select_by(rows: list[dict], score: dict[tuple, float]) -> dict[str, float]:
    by_target: dict[str, list[dict]] = {}
    for r in rows:
        by_target.setdefault(r["target_id"], []).append(r)
    selected: dict[str, float] = {}
    for target, cands in by_target.items():
        best = max(
            cands,
            key=lambda r: (
                score[(r["family"], r["route_id"])],
                -int(r["bridge_count"]),
                r["route_id"],
            ),
        )
        selected[target] = float(best["gt_dice_evaluation_only"])
    return selected


def mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT / "work/kvasir_1pct_anchors/stage1_feature_knn_vitb256")
    parser.add_argument("--tag", type=str, default="lora_p491_e20")
    parser.add_argument("--output-root", type=Path, default=ROOT / "work/kvasir_1pct_anchors/route_selector_vitb256")
    args = parser.parse_args()

    val_rows = []
    test_rows = []
    for key in KEYS:
        val_rows.extend(load_quality(args.root, key, "validation", args.tag))
        test_rows.extend(load_quality(args.root, key, "test", args.tag))
    print(f"quality rows: validation {len(val_rows)}, test {len(test_rows)}", flush=True)

    pools = {
        "P1_b0_b6": BRIDGES,
        "P2_b3_b6": [3, 4, 5, 6],
    }
    report = {"tag": args.tag, "rows": {"validation": len(val_rows), "test": len(test_rows)}}

    for pool_name, bridges in pools.items():
        val_pool = pooled(val_rows, bridges)
        test_pool = pooled(test_rows, bridges)
        if not val_pool or not test_pool:
            print(f"skip {pool_name}: missing rows", flush=True)
            continue
        qm_val = q_multi_map(val_pool)
        qm_test = q_multi_map(test_pool)
        results = {}

        # M1 fixed 0.5/0.5
        for rows, qm, split in ((val_pool, qm_val, "validation"), (test_pool, qm_test, "test")):
            score = {
                (r["family"], r["route_id"]): 0.5 * float(r["q_cycle"]) + 0.5 * qm[(r["family"], r["route_id"])]
                for r in rows
            }
            results[f"M1_0.5_0.5_{split}"] = select_by(rows, score)

        # M2 calibrated linear (grid on validation, apply once to test)
        alpha_grid = [round(a, 2) for a in np.arange(0.0, 1.001, 0.05)]
        candidates = []
        for alpha in alpha_grid:
            score = {
                (r["family"], r["route_id"]): alpha * float(r["q_cycle"])
                + (1.0 - alpha) * qm_val[(r["family"], r["route_id"])]
                for r in val_pool
            }
            sel = select_by(val_pool, score)
            all_scores = [score[(r["family"], r["route_id"])] for r in val_pool]
            all_dice = [float(r["gt_dice_evaluation_only"]) for r in val_pool]
            candidates.append(
                (
                    mean(list(sel.values())),
                    pqr.spearman(all_scores, all_dice),
                    -alpha,
                    alpha,
                )
            )
        best = max(candidates)
        alpha_best = best[3]
        score_test = {
            (r["family"], r["route_id"]): alpha_best * float(r["q_cycle"])
            + (1.0 - alpha_best) * qm_test[(r["family"], r["route_id"])]
            for r in test_pool
        }
        results["M2_calibrated_test"] = select_by(test_pool, score_test)
        results["M2_alpha_best"] = alpha_best
        results["M2_alpha_validation_dice"] = best[0]

        # M3 ridge router (candidate-invariant), fit on validation, apply to test
        scorer = pqr.Ridge.fit(val_pool, ridge=1.0, include_mode=True)
        score_test = {
            (r["family"], r["route_id"]): float(scorer.score(r))
            for r in test_pool
        }
        results["M3_ridge_test"] = select_by(test_pool, score_test)

        # M4 single-signal
        for name, field in (("q_return", "q_cycle"), ("sam_score", "final_sam_score")):
            score = {
                (r["family"], r["route_id"]): float(r.get(field) or 0.0)
                for r in test_pool
            }
            results[f"M4_{name}_test"] = select_by(test_pool, score)
        score = {
            (r["family"], r["route_id"]): qm_test[(r["family"], r["route_id"])]
            for r in test_pool
        }
        results["M4_q_multi_test"] = select_by(test_pool, score)

        # oracle on test pool
        by_target: dict[str, list[dict]] = {}
        for r in test_pool:
            by_target.setdefault(r["target_id"], []).append(r)
        oracle = {
            t: float(max(cands, key=lambda r: r["gt_dice_evaluation_only"])["gt_dice_evaluation_only"])
            for t, cands in by_target.items()
        }

        summary = {}
        for name, sel in results.items():
            if name.endswith("_test"):
                summary[name] = {
                    "selected": mean(list(sel.values())),
                    "oracle": mean(list(oracle.values())),
                    "gap": mean(list(oracle.values())) - mean(list(sel.values())),
                }
            else:
                summary[name] = sel
        report[pool_name] = summary
        print(json.dumps({pool_name: summary}, indent=2, sort_keys=True), flush=True)

    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / f"selector_report_{args.tag}.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("SELECTOR_REPORT_DONE", flush=True)


if __name__ == "__main__":
    main()
