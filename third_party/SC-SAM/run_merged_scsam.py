import argparse
import json
import logging
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloader.TwoStreamBatchSampler import TwoStreamBatchSampler
from dataloader.clinicdb_dataset import ClinicDBDataset
from dataloader.transforms import build_weak_strong_transforms
from trainer_cotraining import Trainer


def parse_args():
    parser = argparse.ArgumentParser(
        description="SC-SAM semi-supervised training with an explicit frozen labeled subset"
    )
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--labeled_list", required=True)
    parser.add_argument("--labeled_num", type=int, default=None)
    parser.add_argument("--labeled_ratio", type=float, default=None)
    parser.add_argument("--split_seed", type=int, default=2026)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--in_channels", type=int, default=3)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--point_nums", type=int, default=5)
    parser.add_argument("--box_nums", type=int, default=1)
    parser.add_argument("--mod", default="sam_adpt")
    parser.add_argument("--model_type", default="vit_b")
    parser.add_argument("--multimask", action="store_true")
    parser.add_argument(
        "--encoder_adapter", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--sam_checkpoint", required=True)
    parser.add_argument("--trained_sam_checkpoint", default="")
    parser.add_argument("--unet_checkpoint", default="")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--UNet_lr", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--labeled_bs", type=int, default=6)
    parser.add_argument("--mixed_iterations", type=int, default=10000)
    parser.add_argument("--max_iterations", type=int, default=40000)
    parser.add_argument("--val_interval", type=int, default=200)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--consistency", type=float, default=0.1)
    parser.add_argument("--consistency_rampup", type=float, default=200.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mode", choices=("train", "test"), default="train")
    parser.add_argument("-thd", type=bool, default=False)
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_dataset(args, split, transforms):
    transform = transforms if split == "train" else transforms["valid_test"]
    return ClinicDBDataset(args, args.data_path, split, transform=transform)


def read_frozen_labeled_paths(path):
    paths = [
        str(Path(line.strip()).resolve())
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not paths:
        raise ValueError("The explicit labeled list is empty")
    if len(paths) != len(set(paths)):
        raise ValueError("The explicit labeled list contains duplicates")
    return paths


def enforce_frozen_labeled_prefix(args, train_dataset):
    frozen_paths = read_frozen_labeled_paths(args.labeled_list)
    by_image = {str(pair[0].resolve()): pair for pair in train_dataset.pairs}
    missing = [path for path in frozen_paths if path not in by_image]
    if missing:
        raise ValueError(f"{len(missing)} frozen labeled paths are absent from train: {missing[:3]}")

    frozen_set = set(frozen_paths)
    labeled_pairs = [by_image[path] for path in frozen_paths]
    unlabeled_pairs = [
        pair for pair in train_dataset.pairs if str(pair[0].resolve()) not in frozen_set
    ]
    train_dataset.pairs = labeled_pairs + unlabeled_pairs
    args.labeled_num = len(labeled_pairs)
    args.labeled_ratio = args.labeled_num / len(train_dataset)
    return frozen_paths


def save_protocol(args, train_dataset, frozen_paths, output_dir):
    actual_labeled = [str(pair[0].resolve()) for pair in train_dataset.pairs[: args.labeled_num]]
    if actual_labeled != frozen_paths:
        raise RuntimeError("Frozen labeled ordering audit failed")
    unlabeled = [
        str(pair[0].resolve()) for pair in train_dataset.pairs[args.labeled_num :]
    ]
    protocol = vars(args).copy()
    protocol.update(
        train_images=len(train_dataset),
        labeled_images=len(actual_labeled),
        unlabeled_images=len(unlabeled),
        labeled_fraction_of_train=len(actual_labeled) / len(train_dataset),
        protocol_kind="explicit_frozen_labeled_list",
    )
    Path(output_dir, "protocol.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8"
    )
    Path(output_dir, "labeled_images.txt").write_text(
        "\n".join(actual_labeled) + "\n", encoding="utf-8"
    )
    Path(output_dir, "unlabeled_images.txt").write_text(
        "\n".join(unlabeled) + "\n", encoding="utf-8"
    )


def train(args):
    transforms = build_weak_strong_transforms(args)
    train_dataset = make_dataset(args, "train", transforms)
    val_dataset = make_dataset(args, "validation", transforms)
    frozen_paths = enforce_frozen_labeled_prefix(args, train_dataset)
    if not 0 < args.labeled_num < len(train_dataset):
        raise ValueError("labeled_num must leave both labeled and unlabeled samples")
    if args.labeled_bs >= args.batch_size:
        raise ValueError("labeled_bs must be smaller than batch_size")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_protocol(args, train_dataset, frozen_paths, output_dir)
    logging.info(
        "SC-SAM explicit protocol: %d frozen labeled + %d unlabeled",
        args.labeled_num,
        len(train_dataset) - args.labeled_num,
    )

    labeled_idxs = list(range(args.labeled_num))
    unlabeled_idxs = list(range(args.labeled_num, len(train_dataset)))
    sampler = TwoStreamBatchSampler(
        labeled_idxs,
        unlabeled_idxs,
        args.batch_size,
        args.batch_size - args.labeled_bs,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        worker_init_fn=lambda worker_id: np.random.seed(args.seed + worker_id),
    )
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1)
    trainer = Trainer(args)

    iteration = 0
    epochs = math.ceil(args.max_iterations / len(train_loader))
    for _ in tqdm(range(epochs), ncols=80):
        for batch in train_loader:
            trainer.train(batch["image"].cuda(), batch["label"].cuda(), iteration)
            iteration += 1
            if iteration % args.val_interval == 0 or iteration == args.max_iterations:
                trainer.val(val_loader, str(output_dir), iteration)
            if iteration >= args.max_iterations:
                sam_final = output_dir / "sam_final_model.pth"
                unet_final = output_dir / "Unet_final_model.pth"
                torch.save(trainer.sam_model.state_dict(), sam_final)
                torch.save(trainer.Unet.state_dict(), unet_final)
                logging.info("saved final SAM checkpoint to %s", sam_final)
                logging.info("saved final UNet checkpoint to %s", unet_final)
                return


def test(args):
    transforms = build_weak_strong_transforms(args)
    test_dataset = make_dataset(args, "test", transforms)
    loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)
    Trainer(args).test(loader)


if __name__ == "__main__":
    args = parse_args()
    seed_everything(args.seed)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=str(Path(args.output_dir) / f"{args.mode}.log"),
        level=logging.INFO,
        format="[%(asctime)s] %(message)s",
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    if args.mode == "train":
        train(args)
    else:
        test(args)
