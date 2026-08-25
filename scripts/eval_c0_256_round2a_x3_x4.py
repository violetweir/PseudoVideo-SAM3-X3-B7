#!/usr/bin/env python3
"""Evaluate validation-selected X3/X4 best and final checkpoints on test."""

from __future__ import annotations

import argparse
import hashlib
import json
from argparse import Namespace
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import run_s27_student as experiment


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_protocol(run_dir: Path) -> Namespace:
    protocol = json.loads((run_dir / "protocol.json").read_text(encoding="utf-8"))
    for key in ("data_path", "labeled_list", "pseudo_manifest", "output_dir"):
        protocol[key] = Path(protocol[key])
    if protocol.get("x0_validation_log"):
        protocol["x0_validation_log"] = Path(protocol["x0_validation_log"])
    return Namespace(**protocol)


def evaluate_checkpoint(run_dir: Path, checkpoint_name: str) -> dict:
    model_args = load_protocol(run_dir)
    transforms = experiment.build_weak_strong_transforms(model_args)
    test_set = experiment.S27Dataset(model_args, "test", transforms["valid_test"])
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=1)
    checkpoint = run_dir / checkpoint_name
    model = experiment.SamUnet(model_args).cuda().eval()
    model.load_state_dict(torch.load(checkpoint, map_location="cuda", weights_only=True))
    result = experiment.evaluate(model, test_loader)
    del model
    torch.cuda.empty_cache()
    return {
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256(checkpoint),
        "test": result,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--x3-run", type=Path, required=True)
    parser.add_argument("--x4-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    runs = {"X3": args.x3_run, "X4": args.x4_run}
    output = {
        "selection_basis": "Each student's validation Dice; test used only after checkpoint freeze",
        "runs": {},
    }
    for name, run_dir in runs.items():
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        output["runs"][name] = {
            "best_validation_dice": summary["best_validation_diagnostic_only"],
            "best_validation_iteration": summary["best_iteration_diagnostic_only"],
            "final_validation": summary["final_validation"],
            "best": evaluate_checkpoint(run_dir, "student_best.pth"),
            "final": evaluate_checkpoint(run_dir, "student_final.pth"),
        }

    x3 = output["runs"]["X3"]
    x4 = output["runs"]["X4"]
    output["x4_minus_x3"] = {
        "best_validation_dice": (
            x4["best_validation_dice"] - x3["best_validation_dice"]
        ),
        "best_test_dice": x4["best"]["test"]["dice"] - x3["best"]["test"]["dice"],
        "final_validation_dice": (
            x4["final_validation"]["dice"] - x3["final_validation"]["dice"]
        ),
        "final_test_dice": x4["final"]["test"]["dice"] - x3["final"]["test"]["dice"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
