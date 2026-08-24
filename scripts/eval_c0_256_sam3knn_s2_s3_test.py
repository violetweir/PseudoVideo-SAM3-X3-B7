#!/usr/bin/env python3
"""Evaluate the frozen S2/S3 best and final checkpoints on test once."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from argparse import Namespace
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCSAM_ROOT = PROJECT_ROOT / "third_party" / "SC-SAM"
for root in (PROJECT_ROOT, SCSAM_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

import run_t24_student as experiment


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    results = {}
    for student in ("S2", "S3"):
        run_dir = args.phase / "students" / student
        protocol = json.loads((run_dir / "protocol.json").read_text())
        for key in ("data_path", "labeled_list", "pseudo_manifest", "output_dir"):
            protocol[key] = Path(protocol[key])
        model_args = Namespace(**protocol)
        transforms = experiment.build_weak_strong_transforms(model_args)
        test_set = experiment.T22StudentDataset(
            model_args, "test", transforms["valid_test"]
        )
        test_loader = DataLoader(
            test_set, batch_size=1, shuffle=False, num_workers=1, pin_memory=True
        )
        model = experiment.SamUnet(model_args).cuda().eval()

        for checkpoint_name, checkpoint_file in (
            ("valbest", "student_best.pth"),
            ("final", "student_final.pth"),
        ):
            checkpoint = run_dir / checkpoint_file
            model.load_state_dict(torch.load(checkpoint, map_location="cuda"))
            with torch.inference_mode():
                metrics = experiment.evaluate(model, test_loader)
            key = f"{student}_{checkpoint_name}"
            results[key] = {
                "student": student,
                "checkpoint_kind": checkpoint_name,
                "checkpoint": str(checkpoint.resolve()),
                "checkpoint_sha256": sha256(checkpoint),
                "test": metrics,
            }
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(
                    {
                        "selection_uses_test_gt": False,
                        "test_evaluation_authorized_by_user": True,
                        "results": results,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            )
            print(json.dumps({"checkpoint": key, **metrics}), flush=True)


if __name__ == "__main__":
    main()
