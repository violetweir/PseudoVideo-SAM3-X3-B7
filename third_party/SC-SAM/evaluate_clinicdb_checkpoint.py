import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataloader.clinicdb_dataset import ClinicDBDataset
from dataloader.transforms import build_weak_strong_transforms
from Model.model import SamUnet
from utils.utils import dice_coef


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--best_iteration", type=int, required=True)
    parser.add_argument("--best_validation_dice", type=float, required=True)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--in_channels", type=int, default=3)
    parser.add_argument("--labeled_num", type=int, default=490)
    parser.add_argument("--split_seed", type=int, default=2026)
    parser.add_argument("--num_workers", type=int, default=1)
    args = parser.parse_args()

    transforms = build_weak_strong_transforms(args)
    test_data = ClinicDBDataset(args, args.data_path, "test", transforms["valid_test"])
    test_loader = DataLoader(
        test_data, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True
    )
    model = SamUnet(args).cuda()
    model.load_state_dict(torch.load(args.checkpoint, map_location="cuda"))
    model.eval()
    values = []
    with torch.inference_mode():
        for batch in test_loader:
            image = batch["image"].cuda(non_blocking=True)
            target = batch["label"].cuda(non_blocking=True)
            _, probabilities = model(image)
            values.append(float(dice_coef(target, probabilities)))
    result = {
        "interim": True,
        "checkpoint": str(Path(args.checkpoint)),
        "best_iteration_at_snapshot": args.best_iteration,
        "best_validation_dice_at_snapshot": args.best_validation_dice,
        "test_images": len(values),
        "test_dice": float(np.mean(values)),
    }
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
