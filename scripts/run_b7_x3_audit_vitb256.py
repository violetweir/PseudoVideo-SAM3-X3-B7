#!/usr/bin/env python3
"""Student-audited route selection (B7 / calibrated linear) on the ViT-B @256
route root, using the existing 256-px X3/S2/S3 student predictions as q_model.

No new student training is required: the phase-1 students were already trained
at 256x256 and their test/validation predictions cover the same 100 Kvasir
targets.  Methods: B7 geometric, q_model-only, and T25-style 3-signal linear
calibration on validation (grid 0.05), all applied once on test.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path("/Data_8TB/lht/PseudoVideo-SAM3-X3-B7")
sys.path.insert(0, str(ROOT / "scripts"))
from analyze_propagation_quality_router import spearman  # noqa: E402

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
BRIDGES = [0, 1, 2, 3, 4, 5, 6]
STUDENTS = ("X3_final", "S2_final", "S3_final")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_mask(path: str | Path, size: int = 256) -> np.ndarray:
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    image = Image.open(p).convert("L")
    if image.size != (size, size):
        image = image.resize((size, size), Image.Resampling.NEAREST)
    return np.asarray(image) > 127


def dice(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(bool), b.astype(bool)
    denom = int(a.sum()) + int(b.sum())
    return 1.0 if denom == 0 else float(2 * np.logical_and(a, b).sum() / denom)


def load_quality(root: Path, key: str, split: str, tag: str) -> list[dict]:
    path = root / key / f"propagation_quality_{split}_{tag}" / "propagation_quality.jsonl"
    rows = read_jsonl(path)
    for row in rows:
        row["family"] = key
    return rows


def student_masks(pred_root: Path, split: str) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for student in STUDENTS:
        path = pred_root / student / f"student_predictions_{split}.jsonl"
        for row in read_jsonl(path):
            out.setdefault((student, row["merged_id"]), load_mask(row["student_binary_mask"]))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT / "work/kvasir_1pct_anchors/stage1_feature_knn_vitb256")
    parser.add_argument("--tag", type=str, default="lora_p491_e20")
    parser.add_argument("--student-root", type=Path, default=ROOT / "work/kvasir_1pct_anchors/phase1/predictions")
    parser.add_argument("--output-root", type=Path, default=ROOT / "work/kvasir_1pct_anchors/route_selector_vitb256")
    args = parser.parse_args()

    quality = {"validation": [], "test": []}
    for split in ("validation", "test"):
        for key in KEYS:
            quality[split].extend(load_quality(args.root, key, split, args.tag))
    smask = {"validation": student_masks(args.student_root, "validation"),
             "test": student_masks(args.student_root, "test")}

    report = {"tag": args.tag}
    for split in ("validation", "test"):
        masks: dict[tuple, np.ndarray] = {}
        for row in quality[split]:
            masks[(row["family"], row["route_id"])] = load_mask(row["forward_mask_path"])
        for row in quality[split]:
            key = (row["family"], row["route_id"])
            target = row["target_id"]
            row["q_model_x3"] = dice(masks[key], smask[split][("X3_final", target)])
            row["q_model_mean"] = float(np.mean([
                dice(masks[key], smask[split][(student, target)])
                for student in STUDENTS
            ]))
            row["_mask"] = masks[key]
        print(f"loaded {split}: {len(quality[split])} rows", flush=True)

    for pool_name, bridges in (("P1_b0_b6", BRIDGES), ("P2_b3_b6", [3, 4, 5, 6])):
        pools = {}
        for split in ("validation", "test"):
            pools[split] = [r for r in quality[split] if int(r["bridge_count"]) in bridges]
            by_target: dict[str, list[dict]] = {}
            for row in pools[split]:
                by_target.setdefault(row["target_id"], []).append(row)
            for target, cands in by_target.items():
                qm = {
                    (row["family"], row["route_id"]): float(np.mean([
                        dice(row["_mask"], other["_mask"])
                        for other in cands
                        if (other["family"], other["route_id"]) != (row["family"], row["route_id"])
                    ]))
                    for row in cands
                }
                for row in cands:
                    row["q_multi"] = qm[(row["family"], row["route_id"])]

        def select(rows: list[dict], scorefn) -> tuple[dict[str, float], dict[str, dict]]:
            by_target = {}
            for row in rows:
                by_target.setdefault(row["target_id"], []).append(row)
            sel, rows_out = {}, {}
            for t, cands in by_target.items():
                best = max(cands, key=lambda r: (scorefn(r), -int(r["bridge_count"]), r["route_id"]))
                sel[t] = float(best["gt_dice_evaluation_only"])
                rows_out[t] = {
                    "family": best["family"],
                    "route_type": best["route_type"],
                    "bridge_count": int(best["bridge_count"]),
                    "dice": float(best["gt_dice_evaluation_only"]),
                    "q_return": float(best["q_cycle"]),
                    "q_multi": float(best["q_multi"]),
                    "q_model_x3": float(best["q_model_x3"]),
                    "q_model_mean": float(best["q_model_mean"]),
                    "route_id": best["route_id"],
                }
            return sel, rows_out

        test_rows = pools["test"]
        by_target_test = {}
        for row in test_rows:
            by_target_test.setdefault(row["target_id"], []).append(row)
        oracle = {
            t: float(max(cands, key=lambda r: r["gt_dice_evaluation_only"])["gt_dice_evaluation_only"])
            for t, cands in by_target_test.items()
        }
        oracle_mean = float(np.mean(list(oracle.values())))

        def summarize(name, sel, rows_out) -> dict:
            vals = list(sel.values())
            out = {"name": name, "selected": float(np.mean(vals)), "oracle": oracle_mean,
                   "gap": oracle_mean - float(np.mean(vals))}
            per_target[name] = rows_out
            return out

        results = []
        per_target: dict[str, dict] = {}

        def b7(r, qmodel_field):
            qr = max(float(r["q_cycle"]), 1e-6)
            qm = max(float(r["q_multi"]), 1e-6)
            qmod = max(float(r[qmodel_field]), 1e-6)
            return (qr * qm ** 2 * qmod ** 2) ** 0.2

        results.append(summarize("B7_X3", *select(test_rows, lambda r: b7(r, "q_model_x3"))))
        results.append(summarize("B7_mean", *select(test_rows, lambda r: b7(r, "q_model_mean"))))
        results.append(summarize("q_model_X3_only", *select(test_rows, lambda r: r["q_model_x3"])))
        results.append(summarize("q_model_mean_only", *select(test_rows, lambda r: r["q_model_mean"])))
        results.append(summarize("q_multi_only", *select(test_rows, lambda r: r["q_multi"])))

        # fixed b6 and oracle per target (for the report)
        fixed_rows: dict[str, dict] = {}
        oracle_rows: dict[str, dict] = {}
        for t, cands in by_target_test.items():
            fixed = [r for r in cands
                     if r["family"] == "anchor_conditioned_target_pooling__knn_cls"
                     and int(r["bridge_count"]) == 6]
            if fixed:
                f = fixed[0]
                fixed_rows[t] = {
                    "family": f["family"], "route_type": f["route_type"],
                    "bridge_count": int(f["bridge_count"]), "dice": float(f["gt_dice_evaluation_only"]),
                    "q_return": float(f["q_cycle"]), "q_multi": float(f["q_multi"]),
                    "q_model_x3": float(f["q_model_x3"]), "q_model_mean": float(f["q_model_mean"]),
                }
            best = max(cands, key=lambda r: r["gt_dice_evaluation_only"])
            oracle_rows[t] = {
                "family": best["family"], "route_type": best["route_type"],
                "bridge_count": int(best["bridge_count"]), "dice": float(best["gt_dice_evaluation_only"]),
                "q_return": float(best["q_cycle"]), "q_multi": float(best["q_multi"]),
                "q_model_x3": float(best["q_model_x3"]), "q_model_mean": float(best["q_model_mean"]),
            }
        per_target["fixed_b6"] = fixed_rows
        per_target["oracle"] = oracle_rows

        # fixed-formula methods on validation (stability reference)
        val_rows = pools["validation"]
        for name, scorefn in (
            ("B7_X3", lambda r: b7(r, "q_model_x3")),
            ("B7_mean", lambda r: b7(r, "q_model_mean")),
            ("q_multi_only", lambda r: r["q_multi"]),
        ):
            sel_val, _ = select(val_rows, scorefn)
            report.setdefault("validation_reference", {}).setdefault(pool_name, {})[name] = float(np.mean(list(sel_val.values())))

        # T25-style 3-signal calibration on validation
        val_rows = pools["validation"]
        by_target_val = {}
        for row in val_rows:
            by_target_val.setdefault(row["target_id"], []).append(row)
        for qmodel_field, label in (("q_model_x3", "X3"), ("q_model_mean", "mean")):
            best = None
            for ir in range(21):
                for im in range(21 - ir):
                    wr, wm = ir * 0.05, im * 0.05
                    ws = round(1.0 - wr - wm, 10)
                    scorefn = lambda r, a=wr, b=wm, c=ws: a * float(r["q_cycle"]) + b * float(r["q_multi"]) + c * float(r[qmodel_field])
                    sel_val, _ = select(val_rows, scorefn)
                    val_dice = float(np.mean(list(sel_val.values())))
                    scores = [scorefn(r) for r in val_rows]
                    gts = [float(r["gt_dice_evaluation_only"]) for r in val_rows]
                    item = (val_dice, spearman(scores, gts), -wr, -wm, -ws, (wr, wm, ws))
                    if best is None or item[:5] > best[:5]:
                        best = item
            wr, wm, ws = best[5]
            scorefn = lambda r, a=wr, b=wm, c=ws: a * float(r["q_cycle"]) + b * float(r["q_multi"]) + c * float(r[qmodel_field])
            results.append(summarize(
                f"lin_cal_{label}",
                *select(test_rows, scorefn),
            ))
            results[-1]["weights"] = {"q_return": wr, "q_multi": wm, "q_model": ws}
            results[-1]["validation_dice"] = best[0]

        # signal spearman on test rows
        gts = [float(r["gt_dice_evaluation_only"]) for r in test_rows]
        for field in ("q_cycle", "q_multi", "q_model_x3", "q_model_mean"):
            sp = spearman([float(r[field]) for r in test_rows], gts)
            print(f"{pool_name} spearman {field}: {sp:.3f}", flush=True)
            report.setdefault("spearman", {})[f"{pool_name}_{field}"] = sp

        report[pool_name] = results
        report[f"{pool_name}_per_target"] = per_target
        for item in results:
            print(json.dumps({pool_name: item}, sort_keys=True), flush=True)

    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / f"b7_student_audit_{args.tag}.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("B7_STUDENT_AUDIT_DONE", flush=True)


if __name__ == "__main__":
    main()
