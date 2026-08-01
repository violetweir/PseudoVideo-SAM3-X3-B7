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
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloader.TwoStreamBatchSampler import TwoStreamBatchSampler
from dataloader.clinicdb_frozen_sam3_dataset import ClinicDBFrozenSAM3Dataset
from dataloader.transforms import build_weak_strong_transforms
from Model.model import SamUnet
from utils.losses import DiceLoss
from utils.utils import dice_coef


def parse_args():
    parser = argparse.ArgumentParser(description="Frozen SAM3 native pseudo-mask teacher")
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--candidate_npz", default="")
    parser.add_argument("--pseudo_dir", default="")
    parser.add_argument(
        "--pseudo_quality_csv",
        default="",
        help="Optional labeled-only calibrated CSV; rows with accepted=0 are excluded.",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--labeled_ratio", type=float, default=0.1)
    parser.add_argument("--labeled_num", type=int, default=None)
    parser.add_argument("--split_seed", type=int, default=2026)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--in_channels", type=int, default=3)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--labeled_bs", type=int, default=6)
    parser.add_argument("--UNet_lr", type=float, default=0.01)
    parser.add_argument("--max_iterations", type=int, default=40000)
    parser.add_argument("--val_interval", type=int, default=200)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--entropy_weight", type=float, default=0.9)
    parser.add_argument(
        "--teacher_label",
        default="",
        help="Accurate artifact label for the offline pseudo-mask source.",
    )
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def entropy_loss(probabilities):
    entropy = -(probabilities * torch.log(probabilities + 1e-6)).sum(dim=1)
    return entropy.mean() / np.log(2.0)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    dices = []
    for batch in loader:
        image = batch["image"].cuda(non_blocking=True)
        target = batch["target"].cuda(non_blocking=True)
        _, probabilities = model(image)
        dices.append(float(dice_coef(target, probabilities)))
    model.train()
    return float(np.mean(dices))


def build_dataset(args, split, transforms):
    transform = transforms if split == "train" else transforms["valid_test"]
    archive = args.candidate_npz if split == "train" else None
    pseudo_dir = args.pseudo_dir if split == "train" else None
    return ClinicDBFrozenSAM3Dataset(
        args, args.data_path, split, transform=transform, candidate_npz=archive, pseudo_dir=pseudo_dir
    )


def save_protocol(args, train_dataset):
    output = Path(args.output_dir)
    labeled_names = [pair[0].name for pair in train_dataset.pairs[: args.labeled_num]]
    unlabeled_names = [pair[0].name for pair in train_dataset.pairs[args.labeled_num :]]
    protocol = vars(args).copy()
    protocol.update(
        teacher=(args.teacher_label or (
            "frozen SAM3 native-score top, full-resolution PNG" if args.pseudo_dir
            else "frozen SAM3 native-score top, 32x32 archive pilot"
        )),
        train_images=len(train_dataset),
        labeled_images=len(labeled_names),
        unlabeled_images=len(unlabeled_names),
        accepted_unlabeled_images=len(train_dataset.accepted_unlabeled_indices),
        rejected_unlabeled_images=(
            len(unlabeled_names) - len(train_dataset.accepted_unlabeled_indices)
        ),
    )
    if train_dataset.pseudo_audit:
        audit = [train_dataset.pseudo_audit[name] for name in unlabeled_names]
        pseudo_dice = np.asarray([item["dice"] for item in audit])
        protocol.update(
            audit_only_native_pseudo_dice_mean=float(pseudo_dice.mean()),
            audit_only_native_pseudo_dice_below_0_1=int((pseudo_dice < 0.1).sum()),
            audit_only_native_pseudo_dice_above_0_8=int((pseudo_dice > 0.8).sum()),
        )
    output.mkdir(parents=True, exist_ok=True)
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    (output / "labeled_images.txt").write_text("\n".join(labeled_names) + "\n", encoding="utf-8")
    (output / "unlabeled_images.txt").write_text("\n".join(unlabeled_names) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    seed_everything(args.seed)
    transforms = build_weak_strong_transforms(args)

    # Determine the count before targets are accessed by workers.
    metadata_count = sum(1 for _ in open(Path(args.data_path) / "train" / "metadata.jsonl"))
    if args.labeled_num is None:
        args.labeled_num = max(1, int(round(metadata_count * args.labeled_ratio)))

    train_dataset = build_dataset(args, "train", transforms)
    val_dataset = build_dataset(args, "validation", transforms)
    test_dataset = build_dataset(args, "test", transforms)
    save_protocol(args, train_dataset)

    labeled = list(range(args.labeled_num))
    unlabeled = train_dataset.accepted_unlabeled_indices
    if not unlabeled:
        raise RuntimeError("Pseudo-quality filter rejected every unlabeled image")
    sampler = TwoStreamBatchSampler(
        labeled, unlabeled, args.batch_size, args.batch_size - args.labeled_bs
    )
    train_loader = DataLoader(
        train_dataset, batch_sampler=sampler, num_workers=args.num_workers,
        pin_memory=True, worker_init_fn=lambda worker_id: np.random.seed(args.seed + worker_id),
    )
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)

    model = SamUnet(args).cuda().train()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.UNet_lr, momentum=0.9, weight_decay=1e-4
    )
    dice_loss = DiceLoss(args.num_classes)
    output = Path(args.output_dir)
    best_val = -1.0
    best_iteration = 0
    iteration = 0
    epochs = math.ceil(args.max_iterations / len(train_loader))

    for _ in tqdm(range(epochs), ncols=80):
        for batch in train_loader:
            image = batch["image"].cuda(non_blocking=True)
            target = batch["target"].cuda(non_blocking=True)
            logits, probabilities = model(image)
            supervised = F.cross_entropy(logits[: args.labeled_bs], target[: args.labeled_bs])
            supervised += dice_loss(probabilities[: args.labeled_bs], target[: args.labeled_bs])
            pseudo = F.cross_entropy(logits[args.labeled_bs :], target[args.labeled_bs :])
            pseudo += dice_loss(probabilities[args.labeled_bs :], target[args.labeled_bs :])
            entropy = entropy_loss(probabilities)
            loss = supervised + pseudo + args.entropy_weight * entropy

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr = args.UNet_lr * (1.0 - iteration / args.max_iterations)
            optimizer.param_groups[0]["lr"] = lr
            iteration += 1
            if iteration == 1 or iteration % 20 == 0:
                logging.info(
                    "iteration %d loss %.6f supervised %.6f pseudo %.6f entropy %.6f lr %.8f",
                    iteration, loss.item(), supervised.item(), pseudo.item(), entropy.item(), lr,
                )
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
    summary = {
        "best_iteration": best_iteration,
        "best_validation_dice": best_val,
        "test_dice": test_dice,
        "teacher": args.teacher_label or (
            "frozen SAM3 native top, full resolution" if args.pseudo_dir else "archive pilot"
        ),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logging.info("FINAL %s", json.dumps(summary))


if __name__ == "__main__":
    args_for_log = parse_args()
    Path(args_for_log.output_dir).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=str(Path(args_for_log.output_dir) / "train.log"), level=logging.INFO,
        format="[%(asctime)s] %(message)s",
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    # Parse once inside main to keep the executable entry point simple.
    main()
