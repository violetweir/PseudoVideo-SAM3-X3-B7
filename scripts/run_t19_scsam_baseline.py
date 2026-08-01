#!/usr/bin/env python3
"""Run the T19 SC-SAM 16GT low-label baseline from the vendored checkout."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sc-sam-root", type=Path, default=ROOT / "third_party/SC-SAM")
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--labeled-list", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sam-checkpoint", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--labeled-bs", type=int, default=6)
    parser.add_argument("--mixed-iterations", type=int, default=10000)
    parser.add_argument("--max-iterations", type=int, default=40000)
    parser.add_argument("--val-interval", type=int, default=200)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--mode", choices=("train", "test"), default="train")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    script = args.sc_sam_root / "run_merged_scsam.py"
    if not script.exists():
        raise FileNotFoundError(script)
    command = [
        args.python,
        "-u",
        script,
        "--data_path",
        args.data_path,
        "--output_dir",
        args.output_dir,
        "--labeled_list",
        args.labeled_list,
        "--sam_checkpoint",
        args.sam_checkpoint,
        "--seed",
        str(args.seed),
        "--split_seed",
        str(args.seed),
        "--batch_size",
        str(args.batch_size),
        "--labeled_bs",
        str(args.labeled_bs),
        "--mixed_iterations",
        str(args.mixed_iterations),
        "--max_iterations",
        str(args.max_iterations),
        "--val_interval",
        str(args.val_interval),
        "--num_workers",
        str(args.num_workers),
        "--mode",
        args.mode,
    ]
    print(" ".join(str(part) for part in command), flush=True)
    if args.dry_run:
        return
    env = os.environ.copy()
    env["SC_SAM_ROOT"] = str(args.sc_sam_root.resolve())
    subprocess.run([str(part) for part in command], cwd=args.sc_sam_root, env=env, check=True)


if __name__ == "__main__":
    main()
