#!/usr/bin/env python3
"""Train the leakage-safe T24 single-image student (S0/S1/S2)."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

# The student is a SC-SAM model, not a SAM3 process.  Make both historical
# project import roots explicit before importing their modules.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCSAM_ROOT = Path(
    os.environ.get("SC_SAM_ROOT", PROJECT_ROOT / "third_party" / "SC-SAM")
).resolve()
for import_root in (PROJECT_ROOT, SCSAM_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import numpy as np
import torch
import torch.nn.functional as F
import albumentations as A
import cv2
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as tv_transforms
from tqdm import tqdm

from dataloader.TwoStreamBatchSampler import TwoStreamBatchSampler
from dataloader.transforms import build_weak_strong_transforms
from Model.model import SamUnet
from utils.losses import DiceLoss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--labeled-list", required=True)
    parser.add_argument("--pseudo-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--experiment", choices=("S0", "S1", "S2", "S3"), required=True
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--in-channels", type=int, default=3)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--labeled-bs", type=int, default=6)
    parser.add_argument("--UNet-lr", type=float, default=0.01)
    parser.add_argument("--max-iterations", type=int, default=40000)
    parser.add_argument("--val-interval", type=int, default=200)
    parser.add_argument("--lambda-pseudo", type=float, default=0.5)
    parser.add_argument("--pseudo-ramp-iterations", type=int, default=2000)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def dataloader_worker_kwargs(num_workers: int) -> dict:
    kwargs = {"num_workers": num_workers, "pin_memory": True}
    if num_workers > 0:
        kwargs.update({"persistent_workers": True, "prefetch_factor": 4})
    return kwargs


def safe_id(value: str) -> str:
    return value.replace("::", "__").replace("/", "_")


def cached_isic_path(split: str, merged_id: str, kind: str) -> Path | None:
    cache_root = os.environ.get("ISIC_RESIZED_CACHE_ROOT")
    if not cache_root:
        return None
    path = Path(cache_root) / split / kind / f"{safe_id(merged_id)}.png"
    return path if path.exists() else None


def maybe_use_light_strong_aug(args: argparse.Namespace, transforms: dict) -> None:
    if os.environ.get("ISIC_LIGHT_STRONG_AUG", "0") != "1":
        return
    transforms["train_strong"] = A.Compose(
        [
            A.Resize(
                args.image_size,
                args.image_size,
                interpolation=cv2.INTER_NEAREST,
                p=1.0,
            ),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.6),
            A.ShiftScaleRotate(
                shift_limit=0.0625,
                scale_limit=0.05,
                rotate_limit=10,
                p=0.5,
            ),
            A.CoarseDropout(
                max_holes=20,
                max_height=args.image_size // 20,
                max_width=args.image_size // 20,
                min_holes=10,
                fill_value=0,
                mask_fill_value=0,
                p=0.7,
            ),
        ],
        p=1.0,
    )


class LongTwoStreamBatchSampler:
    """Two-stream batches with an iteration-sized epoch.

    The upstream sampler defines one epoch as one pass over the tiny labeled
    pool.  With 21 labeled ISIC anchors and labeled_bs=6 that is only three
    batches, which starves DataLoader prefetching.  This sampler keeps the
    same stream composition but emits max_iterations batches per iterator.
    """

    def __init__(
        self,
        primary_indices: list[int],
        secondary_indices: list[int],
        batch_size: int,
        secondary_batch_size: int,
        length: int,
    ) -> None:
        self.primary_indices = list(primary_indices)
        self.secondary_indices = list(secondary_indices)
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size
        self.length = length
        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __len__(self) -> int:
        return self.length

    @staticmethod
    def _eternal(indices: list[int]):
        while True:
            for value in np.random.permutation(indices):
                yield int(value)

    def __iter__(self):
        primary_iter = self._eternal(self.primary_indices)
        secondary_iter = self._eternal(self.secondary_indices)
        for _ in range(self.length):
            batch = [
                next(primary_iter) for _ in range(self.primary_batch_size)
            ] + [
                next(secondary_iter) for _ in range(self.secondary_batch_size)
            ]
            np.random.shuffle(batch)
            yield batch


class T22StudentDataset(Dataset):
    def __init__(self, args: argparse.Namespace, split: str, transform) -> None:
        self.split = split
        self.transform = transform
        records = read_jsonl(Path(args.data_path) / split / "metadata.jsonl")
        pseudo_rows = (
            read_jsonl(Path(args.pseudo_manifest)) if split == "train" else []
        )
        q_values = [float(row["q_multi"]) for row in pseudo_rows]
        q_min, q_max = (min(q_values), max(q_values)) if q_values else (0.0, 1.0)
        pseudo_by_id = {}
        for row in pseudo_rows:
            q = float(row["q_multi"])
            normalized = (q - q_min) / max(q_max - q_min, 1e-12)
            pseudo_by_id[row["target_id"]] = {
                "path": Path(
                    row.get("pseudo_consensus_path", row["pseudo_mask_path"])
                ),
                "pixel_weight_path": (
                    Path(row["pixel_weight_path"])
                    if row.get("pixel_weight_path")
                    else None
                ),
                "q_multi": q,
                "quality_weight": max(0.2, min(1.0, normalized)),
            }
        frozen = {
            str(Path(line.strip()).resolve())
            for line in Path(args.labeled_list).read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        labeled, pseudo, evaluation = [], [], []
        split_root = Path(args.data_path) / split
        for record in records:
            image = Path(record["file_name"])
            gt = Path(record["mask_file_name"])
            if not image.is_absolute():
                image = split_root / image
            if not gt.is_absolute():
                gt = split_root / gt
            image = image.resolve()
            gt = gt.resolve()
            cached_image = cached_isic_path(split, record["merged_id"], "images")
            cached_gt = cached_isic_path(split, record["merged_id"], "masks")
            base = {
                "image": cached_image or image,
                "gt": cached_gt or gt,
                "merged_id": record["merged_id"],
                "source_dataset": record["source_dataset"],
            }
            if split != "train":
                evaluation.append({**base, "target": gt, "is_labeled": True})
            elif str(image) in frozen:
                labeled.append({**base, "target": gt, "is_labeled": True})
            elif record["merged_id"] in pseudo_by_id:
                pseudo_info = pseudo_by_id[record["merged_id"]]
                pseudo.append(
                    {
                        **base,
                        "target": pseudo_info["path"],
                        "is_labeled": False,
                        "q_multi": pseudo_info["q_multi"],
                        "quality_weight": pseudo_info["quality_weight"],
                        "pixel_weight_path": pseudo_info["pixel_weight_path"],
                    }
                )
        if split == "train":
            if not labeled:
                raise RuntimeError("No labeled images found in the frozen list")
            if not pseudo:
                raise RuntimeError("No accepted pseudo images found")
            self.rows = sorted(labeled, key=lambda x: x["merged_id"]) + sorted(
                pseudo, key=lambda x: x["merged_id"]
            )
            self.labeled_count = len(labeled)
        else:
            self.rows = sorted(evaluation, key=lambda x: x["merged_id"])
            self.labeled_count = len(self.rows)
        self.normalize = tv_transforms.Normalize(
            [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        image = np.asarray(Image.open(row["image"]).convert("RGB"), dtype=np.float32) / 255
        mask = np.asarray(Image.open(row["target"]).convert("L"), dtype=np.float32) / 255
        pixel_weight_path = row.get("pixel_weight_path")
        pixel_weight = (
            np.asarray(
                Image.open(pixel_weight_path).convert("L"), dtype=np.float32
            )
            / 255
            if pixel_weight_path
            else np.ones_like(mask, dtype=np.float32)
        )
        if self.transform:
            if self.split == "train":
                key = "train_weak" if row["is_labeled"] else "train_strong"
                transformed = self.transform[key](
                    image=image, masks=[mask, pixel_weight]
                )
            else:
                transformed = self.transform(
                    image=image, masks=[mask, pixel_weight]
                )
            image = transformed["image"]
            mask, pixel_weight = transformed["masks"]
        image_tensor = torch.from_numpy(np.ascontiguousarray(image)).float().permute(2, 0, 1)
        return {
            "image": self.normalize(image_tensor),
            "target": torch.from_numpy(np.ascontiguousarray(mask >= 0.5)).long(),
            "soft_target": torch.from_numpy(
                np.ascontiguousarray(mask)
            ).float(),
            "pixel_weight": torch.from_numpy(
                np.ascontiguousarray(pixel_weight)
            ).float(),
            "is_labeled": row["is_labeled"],
            "quality_weight": float(row.get("quality_weight", 1.0)),
            "merged_id": row["merged_id"],
            "source_dataset": row["source_dataset"],
        }


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader) -> dict:
    model.eval()
    by_dataset: dict[str, list[tuple[float, float, float, float]]] = defaultdict(list)
    for batch in loader:
        image = batch["image"].cuda(non_blocking=True)
        target = batch["target"].cuda(non_blocking=True) > 0
        _, probabilities = model(image)
        prediction = probabilities[:, 1] >= 0.5
        for index, dataset in enumerate(batch["source_dataset"]):
            pred = prediction[index]
            gt = target[index]
            intersection = float((pred & gt).sum())
            pred_area, gt_area = float(pred.sum()), float(gt.sum())
            dice = (2 * intersection + 1e-7) / (pred_area + gt_area + 1e-7)
            iou = (intersection + 1e-7) / (
                pred_area + gt_area - intersection + 1e-7
            )
            by_dataset[dataset].append(
                (dice, iou, float(pred.any()), pred_area / float(pred.numel()))
            )
    model.train()
    all_values = [value for values in by_dataset.values() for value in values]
    return {
        "count": len(all_values),
        "dice": float(np.mean([value[0] for value in all_values])),
        "iou": float(np.mean([value[1] for value in all_values])),
        "nonempty_rate": float(np.mean([value[2] for value in all_values])),
        "mean_area": float(np.mean([value[3] for value in all_values])),
        "by_dataset": {
            dataset: {
                "count": len(values),
                "dice": float(np.mean([value[0] for value in values])),
                "iou": float(np.mean([value[1] for value in values])),
                "nonempty_rate": float(np.mean([value[2] for value in values])),
                "mean_area": float(np.mean([value[3] for value in values])),
            }
            for dataset, values in sorted(by_dataset.items())
        },
    }


def main() -> None:
    args = parse_args()
    # SamUnet expects the historical underscore-style attributes.
    args.num_classes = args.num_classes
    args.in_channels = args.in_channels
    seed_everything(args.seed)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    transforms = build_weak_strong_transforms(args)
    maybe_use_light_strong_aug(args, transforms)
    train_set = T22StudentDataset(args, "train", transforms)
    val_set = T22StudentDataset(args, "validation", transforms["valid_test"])
    test_set = T22StudentDataset(args, "test", transforms["valid_test"])
    labeled = list(range(train_set.labeled_count))
    pseudo = list(range(train_set.labeled_count, len(train_set)))
    if args.experiment == "S0":
        loader = DataLoader(
            torch.utils.data.Subset(train_set, labeled),
            batch_size=min(args.batch_size, len(labeled)),
            shuffle=True,
            drop_last=False,
            **dataloader_worker_kwargs(args.num_workers),
            worker_init_fn=lambda worker_id: np.random.seed(args.seed + worker_id),
        )
    else:
        sampler = LongTwoStreamBatchSampler(
            labeled,
            pseudo,
            args.batch_size,
            args.batch_size - args.labeled_bs,
            args.max_iterations,
        )
        loader = DataLoader(
            train_set,
            batch_sampler=sampler,
            **dataloader_worker_kwargs(args.num_workers),
            worker_init_fn=lambda worker_id: np.random.seed(args.seed + worker_id),
        )
    model = SamUnet(args).cuda().train()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.UNet_lr, momentum=0.9, weight_decay=1e-4
    )
    dice_loss = DiceLoss(args.num_classes)
    protocol = vars(args).copy()
    protocol.update(
        labeled_count=len(labeled),
        pseudo_count=len(pseudo),
        checkpoint_rule=(
            "validate every val_interval; overwrite student_best.pth only when "
            "validation Dice improves; test exactly once after training using best"
        ),
        entropy_weight=0.0,
        pseudo_as_anchor=False,
        experiment=args.experiment,
        pseudo_weight_mode=(
            "none"
            if args.experiment == "S0"
            else (
                "constant"
                if args.experiment == "S1"
                else "normalized_q_multi"
            )
        ),
        quality_normalization="fixed over accepted pseudo labels; clip to [0.2,1.0]",
    )
    (output / "protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    iteration = 0
    best_val = -1.0
    best_iteration = 0
    val_loader = DataLoader(
        val_set, batch_size=1, shuffle=False, **dataloader_worker_kwargs(1)
    )
    test_loader = DataLoader(
        test_set, batch_size=1, shuffle=False, **dataloader_worker_kwargs(1)
    )
    validation_path = output / "validation.jsonl"
    log_path = output / "train.jsonl"
    progress = tqdm(total=args.max_iterations, ncols=80)
    train_iter = iter(loader)
    try:
        while iteration < args.max_iterations:
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(loader)
                batch = next(train_iter)
            image = batch["image"].cuda(non_blocking=True)
            target = batch["target"].cuda(non_blocking=True)
            logits, probabilities = model(image)
            labeled_bs = len(target) if args.experiment == "S0" else args.labeled_bs
            supervised = F.cross_entropy(
                logits[:labeled_bs], target[:labeled_bs]
            ) + dice_loss(probabilities[:labeled_bs], target[:labeled_bs])
            if args.experiment == "S0":
                pseudo_loss = torch.zeros((), device=image.device)
            elif args.experiment in ("S1", "S2"):
                pseudo_logits = logits[labeled_bs:]
                pseudo_target = target[labeled_bs:]
                per_pixel_ce = F.cross_entropy(
                    pseudo_logits, pseudo_target, reduction="none"
                )
                per_sample_ce = per_pixel_ce.flatten(1).mean(1)
                foreground = probabilities[labeled_bs:, 1]
                target_float = (pseudo_target > 0).float()
                intersection = (foreground * target_float).flatten(1).sum(1)
                denominator = (
                    foreground.flatten(1).sum(1)
                    + target_float.flatten(1).sum(1)
                )
                per_sample_dice = 1 - (2 * intersection + 1) / (denominator + 1)
                quality = batch["quality_weight"][labeled_bs:].cuda(
                    non_blocking=True
                )
                if args.experiment == "S1":
                    quality = torch.ones_like(quality)
                pseudo_loss = (
                    quality * (per_sample_ce + per_sample_dice)
                ).mean()
            else:
                foreground = probabilities[labeled_bs:, 1].float()
                soft_target = batch["soft_target"][labeled_bs:].cuda(
                    non_blocking=True
                )
                pixel_weight = batch["pixel_weight"][labeled_bs:].cuda(
                    non_blocking=True
                )
                pixel_bce = F.binary_cross_entropy(
                    foreground.clamp(1e-6, 1 - 1e-6),
                    soft_target,
                    reduction="none",
                )
                per_sample_bce = (pixel_weight * pixel_bce).flatten(1).mean(1)
                intersection = (
                    pixel_weight * foreground * soft_target
                ).flatten(1).sum(1)
                denominator = (
                    (pixel_weight * foreground).flatten(1).sum(1)
                    + (pixel_weight * soft_target).flatten(1).sum(1)
                )
                per_sample_dice = 1 - (2 * intersection + 1) / (
                    denominator + 1
                )
                quality = batch["quality_weight"][labeled_bs:].cuda(
                    non_blocking=True
                )
                pseudo_loss = (
                    quality * (per_sample_bce + per_sample_dice)
                ).mean()
            ramp = min(1.0, iteration / max(1, args.pseudo_ramp_iterations))
            pseudo_weight = (
                0.0 if args.experiment == "S0" else args.lambda_pseudo * ramp
            )
            loss = supervised + pseudo_weight * pseudo_loss
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at iteration {iteration + 1}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            iteration += 1
            progress.update(1)
            lr = args.UNet_lr * (1.0 - iteration / args.max_iterations)
            optimizer.param_groups[0]["lr"] = lr
            if iteration == 1 or iteration % 20 == 0:
                record = {
                    "iteration": iteration,
                    "loss": float(loss.detach()),
                    "supervised": float(supervised.detach()),
                    "pseudo": float(pseudo_loss.detach()),
                    "pseudo_weight": pseudo_weight,
                    "lr": lr,
                }
                with log_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
                logging.info("%s", json.dumps(record))
            if (
                iteration % args.val_interval == 0
                or iteration == args.max_iterations
            ):
                validation = evaluate(model, val_loader)
                validation_record = {
                    "iteration": iteration,
                    **validation,
                    "is_new_best": validation["dice"] > best_val,
                }
                with validation_path.open("a", encoding="utf-8") as handle:
                    handle.write(
                        json.dumps(validation_record, sort_keys=True) + "\n"
                    )
                logging.info("VALIDATION %s", json.dumps(validation_record))
                if validation["dice"] > best_val:
                    best_val = validation["dice"]
                    best_iteration = iteration
                    torch.save(model.state_dict(), output / "student_best.pth")
                    (output / "student_best.json").write_text(
                        json.dumps(
                            {
                                "iteration": best_iteration,
                                "validation": validation,
                                "selection_uses_test_gt": False,
                            },
                            ensure_ascii=False,
                            indent=2,
                        )
                        + "\n",
                        encoding="utf-8",
                    )
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "iteration": iteration,
                        "best_validation_dice": best_val,
                        "best_iteration": best_iteration,
                    },
                    output / "training_latest.pth",
                )
    finally:
        progress.close()
    final_checkpoint = output / "student_final.pth"
    torch.save(model.state_dict(), final_checkpoint)
    # Reconstruct and load the validation-selected checkpoint.  Test is run
    # exactly once here, after all optimization has finished.
    checkpoint = output / "student_best.pth"
    reloaded = SamUnet(args).cuda()
    reloaded.load_state_dict(torch.load(checkpoint, map_location="cuda"))
    model = reloaded
    summary = {
        "final_iteration": iteration,
        "best_iteration": best_iteration,
        "best_validation_dice": best_val,
        "best_validation": evaluate(model, val_loader),
        "test": evaluate(model, test_loader),
        "checkpoint": str(checkpoint),
        "final_checkpoint": str(final_checkpoint),
        "test_evaluations_during_training": 0,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output / "TRAINING_COMPLETE").write_text("complete\n", encoding="utf-8")
    logging.info("FINAL %s", json.dumps(summary))


if __name__ == "__main__":
    parsed = parse_args()
    Path(parsed.output_dir).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=str(Path(parsed.output_dir) / "train.log"),
        level=logging.INFO,
        format="[%(asctime)s] %(message)s",
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    main()
