#!/usr/bin/env python3
"""Run the T20 SynFoC 16GT low-label baseline from the vendored T20 copy."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synfoc-root", type=Path, default=ROOT / "third_party/SynFoC-T20")
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--labeled-list", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--medsam-checkpoint", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-iterations", type=int, default=40000)
    parser.add_argument("--num-eval-iter", type=int, default=500)
    parser.add_argument("--label-bs", type=int, default=4)
    parser.add_argument("--unlabel-bs", type=int, default=4)
    parser.add_argument("--test-bs", type=int, default=8)
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--base-lr", type=float, default=0.03)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    script = args.synfoc_root / "train.py"
    if not script.exists():
        raise FileNotFoundError(script)
    command = [
        args.python,
        "-u",
        script,
        "--dataset",
        "ClinicDB",
        "--dataset_label",
        "Merged CVC-ClinicDB + Kvasir-SEG",
        "--data_path",
        args.data_path,
        "--labeled_list",
        args.labeled_list,
        "--output_dir",
        args.output_dir,
        "--save_name",
        "fixed16_seed2026",
        "--model",
        "MedSAM",
        "--ckpt",
        args.medsam_checkpoint,
        "--max_iterations",
        str(args.max_iterations),
        "--num_eval_iter",
        str(args.num_eval_iter),
        "--label_bs",
        str(args.label_bs),
        "--unlabel_bs",
        str(args.unlabel_bs),
        "--test_bs",
        str(args.test_bs),
        "--img_size",
        str(args.img_size),
        "--base_lr",
        str(args.base_lr),
        "--seed",
        str(args.seed),
        "--AdamW",
        "--warmup",
        "--save_model",
        "--overwrite",
        "--gpu",
        str(args.gpu),
    ]
    print(" ".join(str(part) for part in command), flush=True)
    if args.dry_run:
        return
    subprocess.run([str(part) for part in command], cwd=args.synfoc_root, check=True)


if __name__ == "__main__":
    main()
