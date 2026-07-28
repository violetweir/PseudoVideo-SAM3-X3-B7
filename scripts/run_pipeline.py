#!/usr/bin/env python3
"""Run the S27 X3+B7 reproduction pipeline with resumable stage outputs."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 student environments commonly hit this.
    tomllib = None


ROOT = Path(__file__).resolve().parents[1]


STAGES = (
    "prepare_protocol",
    "t21_round0_train",
    "t21_test_pool0",
    "pseudo568",
    "t24_s2_final",
    "t24_s2_valbest_export",
    "t24_s2_final_export",
    "t24_s3_consensus",
    "t24_s3_final",
    "t24_s3_final_export",
    "s27_remaining_pool",
    "s27_remaining_routes",
    "s27_sam_probabilities",
    "s27_audit_tiers",
    "s27_expansion_sets",
    "s27_x3_train",
    "s27_x3_select",
    "s27_x3_export",
    "validation_routes",
    "t25_b7_validation",
    "t25_b7_test",
)


def load_config(path: Path) -> dict:
    if tomllib is not None:
        with path.open("rb") as handle:
            cfg = tomllib.load(handle)
    else:
        cfg = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.split("#", 1)[0].strip()
            if not stripped or "=" not in stripped:
                continue
            key, raw_value = [part.strip() for part in stripped.split("=", 1)]
            if raw_value.startswith('"') and raw_value.endswith('"'):
                value = raw_value[1:-1]
            elif raw_value.lower() in {"true", "false"}:
                value = raw_value.lower() == "true"
            elif "." in raw_value:
                value = float(raw_value)
            else:
                value = int(raw_value)
            cfg[key] = value
    cfg.setdefault("project_root", str(ROOT))
    cfg.setdefault("work_dir", str(ROOT / "work/reproduction_v1"))
    cfg.setdefault("seed", 2026)
    cfg.setdefault("canvas_size", 512)
    cfg.setdefault("knn_k", 20)
    cfg.setdefault("image_size", 256)
    cfg.setdefault("max_iterations", 40000)
    cfg.setdefault("num_workers", 4)
    return cfg


def run(cmd: list[str], env: dict[str, str], dry_run: bool) -> None:
    print(" ".join(str(part) for part in cmd), flush=True)
    if dry_run:
        return
    subprocess.run([str(part) for part in cmd], cwd=ROOT, env=env, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/reproduction.toml")
    parser.add_argument("--from-stage", choices=STAGES, default=STAGES[0])
    parser.add_argument("--to-stage", choices=STAGES, default=STAGES[-1])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    work = Path(cfg["work_dir"])
    protocol = work / "protocol"
    t21 = work / "t21_dynamic_pseudovideo"
    t22 = work / "pseudo568"
    t24 = work / "committee_students"
    s27 = work / "s27_progressive_x3"
    t25 = work / "b7_route_selection"

    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT), str(ROOT / "src"), env.get("PYTHONPATH", "")]
    )
    if cfg.get("sc_sam_root"):
        env["SC_SAM_ROOT"] = str(cfg["sc_sam_root"])

    sam3_py = cfg["sam3_python"]
    student_py = cfg["student_python"]
    scripts = ROOT / "scripts"
    stage_cmds: dict[str, list[str]] = {
        "prepare_protocol": [
            student_py, scripts / "prepare_protocol.py",
            "--data-root", cfg["data_root"],
            "--splits", ROOT / "protocols/reproduction_v1/splits.jsonl",
            "--support-ids", ROOT / "protocols/reproduction_v1/support_ids.jsonl",
            "--output-dir", protocol,
        ],
        "t21_round0_train": [
            sam3_py, scripts / "run_t21_dynamic_pseudovideo.py",
            "--manifest", protocol / "merged_manifest.jsonl",
            "--support-manifest", protocol / "support_manifest.jsonl",
            "--output-root", t21,
            "--checkpoint", cfg["sam3_checkpoint"],
            "--phase", "round0_train",
            "--resume",
        ],
        "t21_test_pool0": [
            sam3_py, scripts / "run_t21_dynamic_pseudovideo.py",
            "--manifest", protocol / "merged_manifest.jsonl",
            "--support-manifest", protocol / "support_manifest.jsonl",
            "--output-root", t21,
            "--checkpoint", cfg["sam3_checkpoint"],
            "--phase", "test_pool0",
            "--resume",
        ],
        "pseudo568": [
            student_py, scripts / "prepare_t22_training.py",
            "--t21-root", t21,
            "--output-root", t22,
        ],
        "t24_s2_final": [
            student_py, scripts / "run_t24_student.py",
            "--data-path", cfg["data_root"],
            "--labeled-list", protocol / "frozen_labeled_images.txt",
            "--pseudo-manifest", t22 / "protocol/pseudo_train_tau_m090_r095.jsonl",
            "--output-dir", t24 / "S2_final",
            "--experiment", "S2",
            "--max-iterations", str(cfg["max_iterations"]),
            "--num-workers", str(cfg["num_workers"]),
        ],
        "t24_s2_valbest_export": [
            student_py, scripts / "export_t25_student_predictions.py",
            "--run-dir", t24 / "S2_final",
            "--checkpoint", t24 / "S2_final/student_best.pth",
            "--output-root", t24 / "predictions/S2_valbest",
        ],
        "t24_s2_final_export": [
            student_py, scripts / "export_t25_student_predictions.py",
            "--run-dir", t24 / "S2_final",
            "--checkpoint", t24 / "S2_final/student_final.pth",
            "--output-root", t24 / "predictions/S2_final",
        ],
        "t24_s3_consensus": [
            student_py, scripts / "prepare_t24_pseudo_consensus.py",
            "--pseudo-manifest", t22 / "protocol/pseudo_train_tau_m090_r095.jsonl",
            "--route-results", t21 / "round0_train/route_results.jsonl",
            "--output-root", t24 / "S3_consensus",
        ],
        "t24_s3_final": [
            student_py, scripts / "run_t24_student.py",
            "--data-path", cfg["data_root"],
            "--labeled-list", protocol / "frozen_labeled_images.txt",
            "--pseudo-manifest", t24 / "S3_consensus/pseudo_consensus.jsonl",
            "--output-dir", t24 / "S3_final",
            "--experiment", "S3",
            "--max-iterations", str(cfg["max_iterations"]),
            "--num-workers", str(cfg["num_workers"]),
        ],
        "t24_s3_final_export": [
            student_py, scripts / "export_t25_student_predictions.py",
            "--run-dir", t24 / "S3_final",
            "--checkpoint", t24 / "S3_final/student_final.pth",
            "--output-root", t24 / "predictions/S3_final",
        ],
        "s27_remaining_pool": [
            student_py, scripts / "prepare_s27_remaining_pool.py",
            "--train-metadata", Path(cfg["data_root"]) / "train/metadata.jsonl",
            "--human-list", protocol / "frozen_labeled_images.txt",
            "--pseudo568", t22 / "protocol/pseudo_train_tau_m090_r095.jsonl",
            "--output-dir", s27 / "pools",
            "--project-root", ROOT,
        ],
        "s27_remaining_routes": [
            student_py, scripts / "prepare_s27_frozen_routes.py",
            "--remaining", s27 / "pools/unlabeled_remaining.jsonl",
            "--t21-route-results", t21 / "round0_train/route_results.jsonl",
            "--output-dir", s27 / "remaining_routes",
        ],
        "s27_sam_probabilities": [
            sam3_py, scripts / "export_t26_sam3_probabilities.py",
            "--route-results", s27 / "remaining_routes/route_results.jsonl",
            "--split", "train",
            "--output-dir", s27 / "sam_probabilities",
            "--checkpoint", cfg["sam3_checkpoint"],
        ],
        "s27_audit_tiers": [
            student_py, scripts / "build_s27_audit_and_tiers.py",
            "--remaining", s27 / "pools/unlabeled_remaining.jsonl",
            "--routes", s27 / "remaining_routes/route_results.jsonl",
            "--sam-probabilities", s27 / "sam_probabilities/sam3_probabilities_train.jsonl",
            "--pseudo568", s27 / "pools/pseudo568_original.jsonl",
            "--auditor", t24 / "predictions/S2_valbest/student_predictions_train.jsonl",
            "--auditor-name", "S2_valbest",
            "--auditor", t24 / "predictions/S2_final/student_predictions_train.jsonl",
            "--auditor-name", "S2_final",
            "--auditor", t24 / "predictions/S3_final/student_predictions_train.jsonl",
            "--auditor-name", "S3_final",
            "--output-dir", s27 / "audit",
        ],
        "s27_expansion_sets": [
            student_py, scripts / "build_s27_expansion_sets.py",
            "--pseudo568", s27 / "pools/pseudo568_original.jsonl",
            "--tier-a", s27 / "audit/tier_A.jsonl",
            "--tier-b", s27 / "audit/tier_B.jsonl",
            "--tier-c", s27 / "audit/tier_C.jsonl",
            "--output-dir", s27 / "expansion_sets",
        ],
        "s27_x3_train": [
            student_py, scripts / "run_s27_student.py",
            "--data-path", cfg["data_root"],
            "--labeled-list", protocol / "frozen_labeled_images.txt",
            "--pseudo-manifest", s27 / "expansion_sets/X3/pseudo_manifest.jsonl",
            "--output-dir", s27 / "students/X3",
            "--experiment", "X3",
            "--gt-bs", "3", "--original-bs", "3", "--new-bs", "6",
            "--max-iterations", str(cfg["max_iterations"]),
            "--num-workers", str(cfg["num_workers"]),
        ],
        "s27_x3_select": [
            student_py, scripts / "select_and_eval_s27_student.py",
            "--students-dir", s27 / "students",
            "--output-dir", s27 / "selection",
            "--experiments", "X3",
        ],
        "s27_x3_export": [
            student_py, scripts / "export_t25_student_predictions.py",
            "--run-dir", s27 / "students/X3",
            "--checkpoint", s27 / "students/X3/student_final.pth",
            "--output-root", s27 / "predictions/X3_final",
        ],
        "validation_routes": [
            sam3_py, scripts / "run_t25_validation_routes.py",
            "--manifest", protocol / "merged_manifest.jsonl",
            "--support-manifest", protocol / "support_manifest.jsonl",
            "--t21-root", t21,
            "--output-root", t25 / "protocol",
            "--checkpoint", cfg["sam3_checkpoint"],
            "--resume",
        ],
        "t25_b7_validation": [
            student_py, scripts / "run_t25_offline_analysis.py",
            "--routes", t25 / "protocol/validation_pool0/route_results.jsonl",
            "--predictions", s27 / "predictions/X3_final/student_predictions_validation.jsonl",
            "--student", "S27_X3_final",
            "--split", "validation",
            "--output-dir", t25 / "X3_final_B7",
        ],
        "t25_b7_test": [
            student_py, scripts / "run_t25_offline_analysis.py",
            "--routes", t21 / "test_pool0/route_results.jsonl",
            "--predictions", s27 / "predictions/X3_final/student_predictions_test.jsonl",
            "--student", "S27_X3_final",
            "--split", "test",
            "--output-dir", t25 / "X3_final_B7",
        ],
    }

    start = STAGES.index(args.from_stage)
    end = STAGES.index(args.to_stage)
    if start > end:
        raise SystemExit("--from-stage must not be after --to-stage")
    for name in STAGES[start : end + 1]:
        print(f"\n[{name}]", flush=True)
        run(stage_cmds[name], env, args.dry_run)


if __name__ == "__main__":
    main()
