#!/usr/bin/env python3
"""Dry-runnable method ladder from SAM3 single-image baselines to S27 X3+B7.

This runner is intentionally broader than scripts/run_pipeline.py. It includes
early exploratory baselines and diagnostic branches that explain how the final
mainline was reached.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    tomllib = None


ROOT = Path(__file__).resolve().parents[1]
STAGES = (
    "prepare_protocol",
    "b00_single_image_sam3_test",
    "b00_single_image_sam3_val",
    "t18_two_frame_prepare",
    "t18_two_frame_eval",
    "e1_multistep_prepare",
    "e1_star",
    "e1_chain",
    "e1_hybrid",
    "t21_dynamic_three_route",
    "pseudo568",
    "t22_anchor_md_16gt",
    "t22_anchor_mdmem_16gt",
    "t23_single_image_decoder_16gt",
    "t24_committee_students",
    "s27_x3_b7_mainline",
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
            key, raw = [part.strip() for part in stripped.split("=", 1)]
            if raw.startswith('"') and raw.endswith('"'):
                value = raw[1:-1]
            elif raw.lower() in {"true", "false"}:
                value = raw.lower() == "true"
            elif "." in raw:
                value = float(raw)
            else:
                value = int(raw)
            cfg[key] = value
    cfg.setdefault("work_dir", str(ROOT / "work/reproduction_v1"))
    cfg.setdefault("sam3_python", sys.executable)
    cfg.setdefault("student_python", sys.executable)
    cfg.setdefault("sam3_checkpoint", "")
    cfg.setdefault("max_iterations", 40000)
    return cfg


def run(cmd: list[object], env: dict[str, str], dry_run: bool) -> None:
    printable = " ".join(str(part) for part in cmd)
    print(printable, flush=True)
    if not dry_run:
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
    ladder = work / "method_ladder"
    t18 = ladder / "t18_two_frame"
    e1 = ladder / "e1_multistep"
    t21 = work / "t21_dynamic_pseudovideo"
    pseudo568 = work / "pseudo568"
    t22 = ladder / "t22_sam3_adaptation"
    scripts = ROOT / "scripts"

    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT), str(ROOT / "src"), env.get("PYTHONPATH", "")]
    )
    if cfg.get("sc_sam_root"):
        env["SC_SAM_ROOT"] = str(cfg["sc_sam_root"])

    sam3_py = cfg["sam3_python"]
    student_py = cfg["student_python"]
    checkpoint = cfg["sam3_checkpoint"]
    data_root = Path(cfg["data_root"])

    stage_cmds: dict[str, list[object]] = {
        "prepare_protocol": [
            student_py, scripts / "prepare_protocol.py",
            "--data-root", data_root,
            "--splits", ROOT / "protocols/reproduction_v1/splits.jsonl",
            "--support-ids", ROOT / "protocols/reproduction_v1/support_ids.jsonl",
            "--output-dir", protocol,
        ],
        "b00_single_image_sam3_test": [
            sam3_py, scripts / "run_b00_sam3_single_image.py",
            "--manifest", protocol / "merged_manifest.jsonl",
            "--split", "test",
            "--checkpoint", checkpoint,
            "--output-root", ladder / "b00_single_image_sam3",
            "--resume",
        ],
        "b00_single_image_sam3_val": [
            sam3_py, scripts / "run_b00_sam3_single_image.py",
            "--manifest", protocol / "merged_manifest.jsonl",
            "--split", "validation",
            "--checkpoint", checkpoint,
            "--output-root", ladder / "b00_single_image_sam3",
            "--resume",
        ],
        "t18_two_frame_prepare": [
            student_py, scripts / "prepare_t18_full_retrieval.py",
            "--t17-root", work,
            "--output-root", t18,
        ],
        "t18_two_frame_eval": [
            sam3_py, scripts / "eval_t18_pseudovideo_pilot.py",
            "--retrieval-manifest", t18 / "protocol/full_retrieval_manifest.jsonl",
            "--output-root", t18 / "eval",
            "--checkpoint", checkpoint,
            "--resume",
        ],
        "e1_multistep_prepare": [
            student_py, scripts / "prepare_e1_multistep.py",
            "--t18-root", t18,
            "--output-root", e1,
            "--max-depth", "5",
        ],
        "e1_star": [
            sam3_py, scripts / "eval_e1_multistep.py",
            "--manifest", e1 / "protocol/windows.jsonl",
            "--output-root", e1 / "star",
            "--structure", "star",
            "--checkpoint", checkpoint,
            "--split", "test",
            "--min-depth", "1",
            "--max-depth", "5",
            "--resume",
        ],
        "e1_chain": [
            sam3_py, scripts / "eval_e1_multistep.py",
            "--manifest", e1 / "protocol/windows.jsonl",
            "--output-root", e1 / "chain",
            "--structure", "chain",
            "--checkpoint", checkpoint,
            "--split", "test",
            "--min-depth", "1",
            "--max-depth", "5",
            "--resume",
        ],
        "e1_hybrid": [
            sam3_py, scripts / "eval_e1_multistep.py",
            "--manifest", e1 / "protocol/windows.jsonl",
            "--output-root", e1 / "hybrid",
            "--structure", "hybrid",
            "--checkpoint", checkpoint,
            "--split", "test",
            "--min-depth", "1",
            "--max-depth", "5",
            "--resume",
        ],
        "t21_dynamic_three_route": [
            sam3_py, scripts / "run_t21_dynamic_pseudovideo.py",
            "--manifest", protocol / "merged_manifest.jsonl",
            "--support-manifest", protocol / "support_manifest.jsonl",
            "--output-root", t21,
            "--checkpoint", checkpoint,
            "--phase", "round0_train",
            "--resume",
        ],
        "pseudo568": [
            student_py, scripts / "prepare_t22_training.py",
            "--t21-root", t21,
            "--output-root", pseudo568,
        ],
        "t22_anchor_md_16gt": [
            sam3_py, scripts / "train_t22_sam3_tracker.py",
            "--manifest", pseudo568 / "protocol/human16_anchors.jsonl",
            "--base-checkpoint", checkpoint,
            "--output-dir", t22 / "anchor_md",
            "--mode", "anchor",
            "--variant", "md",
            "--steps", "400",
            "--save-step0",
        ],
        "t22_anchor_mdmem_16gt": [
            sam3_py, scripts / "train_t22_sam3_tracker.py",
            "--manifest", pseudo568 / "protocol/human16_anchors.jsonl",
            "--base-checkpoint", checkpoint,
            "--output-dir", t22 / "anchor_mdmem",
            "--mode", "anchor",
            "--variant", "md_mem",
            "--steps", "400",
            "--save-step0",
        ],
        "t23_single_image_decoder_16gt": [
            sam3_py, scripts / "train_t23_single_image_decoder.py",
            "--gt-video-manifest", pseudo568 / "protocol/human16_anchors.jsonl",
            "--pseudo-manifest", pseudo568 / "protocol/pseudo_train_tau_m090_r095.jsonl",
            "--base-checkpoint", checkpoint,
            "--output-dir", t22 / "single_image_decoder",
            "--steps", "400",
        ],
        "t24_committee_students": [
            student_py, scripts / "run_pipeline.py",
            "--config", args.config,
            "--from-stage", "t24_s2_final",
            "--to-stage", "s27_expansion_sets",
        ],
        "s27_x3_b7_mainline": [
            student_py, scripts / "run_pipeline.py",
            "--config", args.config,
            "--from-stage", "s27_x3_train",
            "--to-stage", "t25_b7_test",
        ],
    }

    start = STAGES.index(args.from_stage)
    end = STAGES.index(args.to_stage)
    if start > end:
        raise SystemExit("--from-stage must not be after --to-stage")
    for stage in STAGES[start : end + 1]:
        print(f"\n[{stage}]", flush=True)
        run(stage_cmds[stage], env, args.dry_run)


if __name__ == "__main__":
    main()
