import argparse
import json
import logging
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from dataloader.clinicdb_dataset import ClinicDBDataset
from dataloader.transforms import build_weak_strong_transforms
from Model.model import SamUnet
from utils.losses import DiceLoss
from utils.utils import dice_coef


def parse_args():
    parser = argparse.ArgumentParser(description="Strict labeled-only ClinicDB baseline")
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--labeled_num", type=int, default=49)
    parser.add_argument("--split_seed", type=int, default=2026)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--in_channels", type=int, default=3)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--UNet_lr", type=float, default=0.01)
    parser.add_argument("--max_iterations", type=int, default=40000)
    parser.add_argument("--val_interval", type=int, default=200)
    parser.add_argument("--num_workers", type=int, default=4)
    return parser.parse_args()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    values = []
    for batch in loader:
        _, probabilities = model(batch["image"].cuda(non_blocking=True))
        values.append(float(dice_coef(batch["label"].cuda(non_blocking=True), probabilities)))
    model.train()
    return float(np.mean(values))


def main():
    args = parse_args()
    seed_all(args.seed)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    transforms = build_weak_strong_transforms(args)
    train_all = ClinicDBDataset(args, args.data_path, "train", transforms)
    train_data = Subset(train_all, range(args.labeled_num))
    val_data = ClinicDBDataset(args, args.data_path, "validation", transforms["valid_test"])
    test_data = ClinicDBDataset(args, args.data_path, "test", transforms["valid_test"])
    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, drop_last=True,
        num_workers=args.num_workers, pin_memory=True,
        worker_init_fn=lambda worker_id: np.random.seed(args.seed + worker_id),
    )
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=1)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=1)
    labeled_names = [pair[0].name for pair in train_all.pairs[: args.labeled_num]]
    (output / "labeled_images.txt").write_text("\n".join(labeled_names) + "\n", encoding="utf-8")
    (output / "protocol.json").write_text(
        json.dumps({**vars(args), "unlabeled_images_used": 0}, indent=2), encoding="utf-8"
    )

    model = SamUnet(args).cuda().train()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.UNet_lr, momentum=0.9, weight_decay=1e-4)
    dice_loss = DiceLoss(args.num_classes)
    best_val, best_iteration = -1.0, 0
    iteration = 0
    epochs = math.ceil(args.max_iterations / len(train_loader))
    for _ in tqdm(range(epochs), ncols=80):
        for batch in train_loader:
            image = batch["image"].cuda(non_blocking=True)
            target = batch["label"].cuda(non_blocking=True)
            logits, probabilities = model(image)
            loss = F.cross_entropy(logits, target) + dice_loss(probabilities, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr = args.UNet_lr * (1.0 - iteration / args.max_iterations)
            optimizer.param_groups[0]["lr"] = lr
            iteration += 1
            if iteration == 1 or iteration % 20 == 0:
                logging.info("iteration %d loss %.6f lr %.8f", iteration, loss.item(), lr)
            if iteration % args.val_interval == 0 or iteration == args.max_iterations:
                val_dice = evaluate(model, val_loader)
                logging.info("iteration %d val_dice %.6f", iteration, val_dice)
                if val_dice > best_val:
                    best_val, best_iteration = val_dice, iteration
                    torch.save(model.state_dict(), output / "student_best.pth")
            if iteration >= args.max_iterations:
                break
        if iteration >= args.max_iterations:
            break
    model.load_state_dict(torch.load(output / "student_best.pth", map_location="cuda"))
    test_dice = evaluate(model, test_loader)
    summary = {"best_iteration": best_iteration, "best_validation_dice": best_val, "test_dice": test_dice}
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logging.info("FINAL %s", json.dumps(summary))


if __name__ == "__main__":
    args = parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=str(Path(args.output_dir) / "train.log"), level=logging.INFO,
        format="[%(asctime)s] %(message)s",
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    main()
