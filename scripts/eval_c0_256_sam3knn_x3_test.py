#!/usr/bin/env python3
"""Evaluate frozen X3 val-best and final checkpoints on test."""

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

import run_s27_student as experiment


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    protocol = json.loads((args.run_dir / "protocol.json").read_text())
    for key in ("data_path", "labeled_list", "pseudo_manifest", "output_dir"):
        protocol[key] = Path(protocol[key])
    model_args = Namespace(**protocol)
    transforms = experiment.build_weak_strong_transforms(model_args)
    test_set = experiment.S27Dataset(model_args, "test", transforms["valid_test"])
    test_loader = DataLoader(
        test_set, batch_size=1, shuffle=False, num_workers=1, pin_memory=True
    )
    model = experiment.SamUnet(model_args).cuda().eval()

    results = {}
    for kind, filename in (("valbest", "student_best.pth"), ("final", "student_final.pth")):
        checkpoint = args.run_dir / filename
        model.load_state_dict(torch.load(checkpoint, map_location="cuda"))
        with torch.inference_mode():
            metrics = experiment.evaluate(model, test_loader)
        results[kind] = {
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": sha256(checkpoint),
            "test": metrics,
        }
        print(json.dumps({"checkpoint": kind, **metrics}), flush=True)

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


if __name__ == "__main__":
    main()
