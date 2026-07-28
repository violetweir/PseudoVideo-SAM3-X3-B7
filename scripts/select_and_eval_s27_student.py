#!/usr/bin/env python3
"""Select S27 by Final Validation only, freeze it, then evaluate Test once."""

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
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--students-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--experiments", nargs="+", default=["X0", "X1", "X2", "X3", "X4"])
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selection_path = args.output_dir / "frozen_student_selection.json"
    test_path = args.output_dir / "selected_student_test.json"

    validation = {}
    for name in args.experiments:
        run = args.students_dir / name
        summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
        validation[name] = summary["final_validation"]
    selected = max(args.experiments, key=lambda name: (validation[name]["dice"], name))
    run_dir = args.students_dir / selected
    checkpoint = run_dir / "student_final.pth"
    frozen = {
        "selection_basis": "Final-40k Validation Dice only",
        "selected_experiment": selected,
        "validation_results": validation,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256(checkpoint),
        "test_gt_used_for_selection": False,
    }
    if selection_path.exists():
        existing = json.loads(selection_path.read_text(encoding="utf-8"))
        if existing != frozen:
            raise RuntimeError("Frozen student selection already exists and differs")
    else:
        selection_path.write_text(
            json.dumps(frozen, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    if test_path.exists():
        print(test_path.read_text(encoding="utf-8"), end="")
        return

    protocol = json.loads((run_dir / "protocol.json").read_text(encoding="utf-8"))
    for key in ("data_path", "labeled_list", "pseudo_manifest", "output_dir"):
        protocol[key] = Path(protocol[key])
    model_args = Namespace(**protocol)
    transforms = experiment.build_weak_strong_transforms(model_args)
    test_set = experiment.S27Dataset(
        model_args, "test", transforms["valid_test"]
    )
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=1)
    model = experiment.SamUnet(model_args).cuda().eval()
    model.load_state_dict(torch.load(checkpoint, map_location="cuda", weights_only=True))
    result = {
        **frozen,
        "test": experiment.evaluate(model, test_loader),
        "test_evaluation_count_after_freeze": 1,
    }
    test_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
