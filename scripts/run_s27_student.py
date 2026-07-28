#!/usr/bin/env python3
"""Train an S27 three-stream student without evaluating Test during selection."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCSAM_ROOT = Path(
    os.environ.get("SC_SAM_ROOT", PROJECT_ROOT / "third_party" / "SC-SAM")
).resolve()
for root in (PROJECT_ROOT, SCSAM_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision import transforms as tv_transforms

from dataloader.transforms import build_weak_strong_transforms
from Model.model import SamUnet


TYPE_ID = {"gt": 0, "original": 1, "tier_a": 2, "tier_b": 3}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--labeled-list", type=Path, required=True)
    parser.add_argument("--pseudo-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--in-channels", type=int, default=3)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--gt-bs", type=int, required=True)
    parser.add_argument("--original-bs", type=int, required=True)
    parser.add_argument("--new-bs", type=int, required=True)
    parser.add_argument("--UNet-lr", type=float, default=0.01)
    parser.add_argument("--max-iterations", type=int, default=40000)
    parser.add_argument("--val-interval", type=int, default=200)
    parser.add_argument("--grad-monitor-interval", type=int, default=500)
    parser.add_argument("--lambda-pseudo-global", type=float, default=0.5)
    parser.add_argument("--lambda-original", type=float, default=1.0)
    parser.add_argument("--lambda-tier-a", type=float, default=0.75)
    parser.add_argument("--lambda-tier-b", type=float, default=0.50)
    parser.add_argument("--pseudo-ramp-iterations", type=int, default=2000)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--small-area-max", type=float, default=0.059459686279296875)
    parser.add_argument("--medium-area-max", type=float, default=0.1153411865234375)
    parser.add_argument("--x0-validation-log", type=Path)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def decode_probability(path: str | Path) -> np.ndarray:
    array = np.asarray(Image.open(path))
    if array.dtype == np.uint16:
        return array.astype(np.float32) / 65535.0
    if np.issubdtype(array.dtype, np.floating):
        return np.clip(array.astype(np.float32), 0.0, 1.0)
    return array.astype(np.float32) / 255.0


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class ThreeStreamBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        streams: dict[str, list[int]],
        counts: dict[str, int],
        seed: int,
    ) -> None:
        self.streams = streams
        self.counts = counts
        self.seed = seed
        for name, count in counts.items():
            if count > 0 and not streams[name]:
                raise RuntimeError(f"Nonzero batch count for empty stream {name}")
        if sum(counts.values()) <= 0:
            raise RuntimeError("Empty batch")
        self.length = max(
            math.ceil(len(streams[name]) / count)
            for name, count in counts.items()
            if count > 0
        )
        self.epoch = 0

    def __len__(self) -> int:
        return self.length

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        def eternal(pool: list[int]):
            while True:
                for value in rng.permutation(pool):
                    yield int(value)
        iterators = {
            name: eternal(pool) if self.counts[name] else None
            for name, pool in self.streams.items()
        }
        for _ in range(self.length):
            batch = []
            for name in ("gt", "original", "new"):
                if self.counts[name]:
                    batch.extend(
                        next(iterators[name]) for _ in range(self.counts[name])
                    )
            rng.shuffle(batch)
            yield batch


class S27Dataset(Dataset):
    def __init__(self, args: argparse.Namespace, split: str, transform) -> None:
        self.split = split
        self.transform = transform
        metadata = read_jsonl(args.data_path / split / "metadata.jsonl")
        frozen_paths = {
            str(Path(line.strip()).resolve())
            for line in args.labeled_list.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        pseudo_by_id = {
            row["target_id"]: row for row in read_jsonl(args.pseudo_manifest)
        } if split == "train" else {}
        rows = []
        for record in metadata:
            image = Path(record["file_name"]).resolve()
            base = {
                "image": image,
                "gt": Path(record["mask_file_name"]).resolve(),
                "merged_id": record["merged_id"],
                "source_dataset": record["source_dataset"],
            }
            if split != "train":
                rows.append(
                    {
                        **base,
                        "target": base["gt"],
                        "pixel_weight": None,
                        "sample_type": "gt",
                        "image_weight": 1.0,
                    }
                )
            elif str(image) in frozen_paths:
                rows.append(
                    {
                        **base,
                        "target": base["gt"],
                        "pixel_weight": None,
                        "sample_type": "gt",
                        "image_weight": 1.0,
                    }
                )
            elif record["merged_id"] in pseudo_by_id:
                pseudo = pseudo_by_id[record["merged_id"]]
                rows.append(
                    {
                        **base,
                        "target": Path(
                            pseudo.get(
                                "pseudo_consensus_path", pseudo["pseudo_mask_path"]
                            )
                        ),
                        "pixel_weight": (
                            Path(pseudo["pixel_weight_path"])
                            if pseudo.get("pixel_weight_path")
                            else None
                        ),
                        "sample_type": pseudo["sample_type"],
                        "image_weight": float(pseudo["explicit_quality_weight"]),
                        "q_multi": float(pseudo.get("q_multi", math.nan)),
                        "q_model_mean": float(pseudo.get("q_model_mean", math.nan)),
                        "q_model_var": float(pseudo.get("q_model_var", math.nan)),
                    }
                )
        self.rows = sorted(rows, key=lambda row: (TYPE_ID[row["sample_type"]], row["merged_id"]))
        if split == "train":
            counts = Counter(row["sample_type"] for row in self.rows)
            if counts["gt"] != 16 or counts["original"] != 568:
                raise RuntimeError(f"Invalid fixed pools: {dict(counts)}")
        self.normalize = tv_transforms.Normalize(
            [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        image = np.asarray(Image.open(row["image"]).convert("RGB"), dtype=np.float32) / 255
        target = decode_probability(row["target"])
        pixel_weight = (
            decode_probability(row["pixel_weight"])
            if row["pixel_weight"]
            else np.ones_like(target, dtype=np.float32)
        )
        if row["sample_type"] == "tier_a":
            pixel_weight = np.ones_like(target, dtype=np.float32)
        elif row["sample_type"] == "tier_b":
            # Preregistered strong/weak/ignore mapping.
            pixel_weight = np.where(
                pixel_weight >= 0.70,
                1.0,
                np.where(pixel_weight >= 0.30, 0.25, 0.0),
            ).astype(np.float32)
        if self.transform:
            if self.split == "train":
                key = "train_weak" if row["sample_type"] == "gt" else "train_strong"
                transformed = self.transform[key](
                    image=image, masks=[target, pixel_weight]
                )
            else:
                transformed = self.transform(
                    image=image, masks=[target, pixel_weight]
                )
            image = transformed["image"]
            target, pixel_weight = transformed["masks"]
        image_tensor = torch.from_numpy(np.ascontiguousarray(image)).float().permute(2, 0, 1)
        return {
            "image": self.normalize(image_tensor),
            "target": torch.from_numpy(np.ascontiguousarray(target >= 0.5)).long(),
            "soft_target": torch.from_numpy(np.ascontiguousarray(target)).float(),
            "pixel_weight": torch.from_numpy(np.ascontiguousarray(pixel_weight)).float(),
            "sample_type_id": TYPE_ID[row["sample_type"]],
            "sample_type": row["sample_type"],
            "image_weight": row["image_weight"],
            "merged_id": row["merged_id"],
            "source_dataset": row["source_dataset"],
            "q_multi": row.get("q_multi", math.nan),
            "q_model_mean": row.get("q_model_mean", math.nan),
            "q_model_var": row.get("q_model_var", math.nan),
        }


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    small_area_max: float = 0.059459686279296875,
    medium_area_max: float = 0.1153411865234375,
) -> dict:
    model.eval()
    values: dict[str, list[tuple[float, float, float, float]]] = defaultdict(list)
    for batch in loader:
        image = batch["image"].cuda(non_blocking=True)
        target = batch["target"].cuda(non_blocking=True) > 0
        _, probability = model(image)
        prediction = probability[:, 1] >= 0.5
        for index, dataset in enumerate(batch["source_dataset"]):
            pred, gt = prediction[index], target[index]
            intersection = float((pred & gt).sum())
            pred_area, gt_area = float(pred.sum()), float(gt.sum())
            dice = (2 * intersection + 1e-7) / (pred_area + gt_area + 1e-7)
            iou = (intersection + 1e-7) / (
                pred_area + gt_area - intersection + 1e-7
            )
            values[dataset].append(
                (
                    dice,
                    iou,
                    float(pred.any()),
                    pred_area / pred.numel(),
                    gt_area / gt.numel(),
                )
            )
    all_values = [item for group in values.values() for item in group]
    model.train()
    return {
        "count": len(all_values),
        "dice": float(np.mean([x[0] for x in all_values])),
        "iou": float(np.mean([x[1] for x in all_values])),
        "nonempty_rate": float(np.mean([x[2] for x in all_values])),
        "mean_area": float(np.mean([x[3] for x in all_values])),
        "by_dataset": {
            name: {
                "count": len(group),
                "dice": float(np.mean([x[0] for x in group])),
                "iou": float(np.mean([x[1] for x in group])),
                "nonempty_rate": float(np.mean([x[2] for x in group])),
                "mean_area": float(np.mean([x[3] for x in group])),
            }
            for name, group in sorted(values.items())
        },
        "by_size": {
            size: {
                "count": len(group),
                "dice": float(np.mean([x[0] for x in group])) if group else None,
                "iou": float(np.mean([x[1] for x in group])) if group else None,
            }
            for size, group in (
                (
                    "small",
                    [x for x in all_values if x[4] <= small_area_max],
                ),
                (
                    "medium",
                    [
                        x
                        for x in all_values
                        if small_area_max < x[4] <= medium_area_max
                    ],
                ),
                (
                    "large",
                    [x for x in all_values if x[4] > medium_area_max],
                ),
            )
        },
    }


def weighted_soft_loss(
    probability: torch.Tensor,
    target: torch.Tensor,
    pixel_weight: torch.Tensor,
    image_weight: torch.Tensor,
) -> torch.Tensor:
    bce = F.binary_cross_entropy(
        probability.clamp(1e-6, 1 - 1e-6), target, reduction="none"
    )
    weight_sum = pixel_weight.flatten(1).sum(1).clamp_min(1.0)
    per_bce = (pixel_weight * bce).flatten(1).sum(1) / weight_sum
    intersection = (pixel_weight * probability * target).flatten(1).sum(1)
    denominator = (
        (pixel_weight * probability).flatten(1).sum(1)
        + (pixel_weight * target).flatten(1).sum(1)
    )
    per_dice = 1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)
    return (image_weight * (per_bce + per_dice)).mean()


def hard_loss(
    logits: torch.Tensor,
    probability: torch.Tensor,
    target: torch.Tensor,
    image_weight: torch.Tensor,
) -> torch.Tensor:
    per_ce = F.cross_entropy(logits, target, reduction="none").flatten(1).mean(1)
    foreground = probability[:, 1]
    target_float = (target > 0).float()
    intersection = (foreground * target_float).flatten(1).sum(1)
    denominator = foreground.flatten(1).sum(1) + target_float.flatten(1).sum(1)
    per_dice = 1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)
    return (image_weight * (per_ce + per_dice)).mean()


def gradient_norm(loss: torch.Tensor, model: torch.nn.Module) -> float:
    if not loss.requires_grad:
        return 0.0
    gradients = torch.autograd.grad(
        loss, [p for p in model.parameters() if p.requires_grad],
        retain_graph=True, allow_unused=True
    )
    total = sum(float(g.detach().float().pow(2).sum()) for g in gradients if g is not None)
    return math.sqrt(total)


def main() -> None:
    args = parse_args()
    if args.gt_bs + args.original_bs + args.new_bs != args.batch_size:
        raise RuntimeError("Three stream counts must sum to batch size")
    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    transforms = build_weak_strong_transforms(args)
    train_set = S27Dataset(args, "train", transforms)
    val_set = S27Dataset(args, "validation", transforms["valid_test"])
    streams = {
        "gt": [i for i, row in enumerate(train_set.rows) if row["sample_type"] == "gt"],
        "original": [
            i for i, row in enumerate(train_set.rows) if row["sample_type"] == "original"
        ],
        "new": [
            i for i, row in enumerate(train_set.rows)
            if row["sample_type"] in ("tier_a", "tier_b")
        ],
    }
    sampler = ThreeStreamBatchSampler(
        streams,
        {"gt": args.gt_bs, "original": args.original_bs, "new": args.new_bs},
        args.seed,
    )
    train_loader = DataLoader(
        train_set,
        batch_sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        worker_init_fn=lambda worker_id: np.random.seed(args.seed + worker_id),
    )
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=1)
    def evaluate_validation(current_model: torch.nn.Module) -> dict:
        return evaluate(
            current_model,
            val_loader,
            small_area_max=args.small_area_max,
            medium_area_max=args.medium_area_max,
        )
    model = SamUnet(args).cuda().train()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.UNet_lr, momentum=0.9, weight_decay=1e-4
    )
    protocol = {
        **vars(args),
        "data_path": str(args.data_path),
        "labeled_list": str(args.labeled_list),
        "pseudo_manifest": str(args.pseudo_manifest),
        "output_dir": str(args.output_dir),
        "x0_validation_log": (
            str(args.x0_validation_log) if args.x0_validation_log else None
        ),
        "stream_sizes": {name: len(indices) for name, indices in streams.items()},
        "loss_interpretation": (
            "L_gt + 0.5*ramp*(1.0*L_original + 0.75*L_A + 0.50*L_B); "
            "preserves T24 S2 global pseudo coefficient"
        ),
        "checkpoint_selection": "Final-40k Validation only; Test not evaluated",
    }
    (args.output_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    iteration, best_val, best_iteration = 0, -1.0, 0
    latest = args.output_dir / "training_latest.pth"
    if args.resume and latest.exists():
        state = torch.load(latest, map_location="cuda")
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        iteration = int(state["iteration"])
        best_val = float(state["best_validation_dice"])
        best_iteration = int(state["best_iteration"])
        random.setstate(state["python_rng_state"])
        np.random.set_state(state["numpy_rng_state"])
        torch.set_rng_state(state["torch_rng_state"].cpu())
        torch.cuda.set_rng_state_all(state["cuda_rng_state_all"])

    train_log = args.output_dir / "train.jsonl"
    val_log = args.output_dir / "validation.jsonl"
    x0_validation = {}
    if args.x0_validation_log and args.x0_validation_log.exists():
        x0_validation = {
            int(row["iteration"]): float(row["dice"])
            for row in read_jsonl(args.x0_validation_log)
        }
    safety_streak = 0
    while iteration < args.max_iterations:
        for batch in train_loader:
            image = batch["image"].cuda(non_blocking=True)
            target = batch["target"].cuda(non_blocking=True)
            soft = batch["soft_target"].cuda(non_blocking=True)
            pixel = batch["pixel_weight"].cuda(non_blocking=True)
            image_weight = batch["image_weight"].float().cuda(non_blocking=True)
            sample_type = batch["sample_type_id"].cuda(non_blocking=True)
            logits, probabilities = model(image)
            zero = logits.sum() * 0.0
            components = {}
            mask = sample_type == TYPE_ID["gt"]
            components["gt"] = (
                hard_loss(
                    logits[mask], probabilities[mask], target[mask],
                    torch.ones_like(image_weight[mask])
                ) if mask.any() else zero
            )
            mask = sample_type == TYPE_ID["original"]
            components["original"] = (
                hard_loss(
                    logits[mask], probabilities[mask], target[mask], image_weight[mask]
                ) if mask.any() else zero
            )
            for name, type_name in (("tier_a", "tier_a"), ("tier_b", "tier_b")):
                mask = sample_type == TYPE_ID[type_name]
                components[name] = (
                    weighted_soft_loss(
                        probabilities[mask, 1], soft[mask], pixel[mask],
                        image_weight[mask]
                    ) if mask.any() else zero
                )
            ramp = min(1.0, iteration / max(1, args.pseudo_ramp_iterations))
            pseudo = (
                args.lambda_original * components["original"]
                + args.lambda_tier_a * components["tier_a"]
                + args.lambda_tier_b * components["tier_b"]
            )
            loss = components["gt"] + args.lambda_pseudo_global * ramp * pseudo
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at iteration {iteration + 1}")
            grad_record = {}
            if iteration == 0 or (iteration + 1) % args.grad_monitor_interval == 0:
                grad_record = {
                    f"grad_norm_{name}": gradient_norm(value, model)
                    for name, value in components.items()
                }
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=100.0)
            optimizer.step()
            iteration += 1
            lr = args.UNet_lr * (1.0 - iteration / args.max_iterations)
            optimizer.param_groups[0]["lr"] = lr
            if iteration == 1 or iteration % 20 == 0 or grad_record:
                record = {
                    "iteration": iteration,
                    "loss": float(loss.detach()),
                    "gt_loss": float(components["gt"].detach()),
                    "original_loss": float(components["original"].detach()),
                    "tier_a_loss": float(components["tier_a"].detach()),
                    "tier_b_loss": float(components["tier_b"].detach()),
                    "ramp": ramp,
                    "lr": lr,
                    "batch_type_counts": dict(Counter(batch["sample_type"])),
                    **grad_record,
                }
                with train_log.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
            if iteration % args.val_interval == 0 or iteration == args.max_iterations:
                validation = evaluate_validation(model)
                safety_flags = {
                    "low_nonempty": validation["nonempty_rate"] < 0.50,
                    "area_collapse": validation["mean_area"] < 0.01,
                    "below_x0_by_0_05": (
                        iteration in x0_validation
                        and validation["dice"] < x0_validation[iteration] - 0.05
                    ),
                }
                if iteration >= 2000 and any(safety_flags.values()):
                    safety_streak += 1
                else:
                    safety_streak = 0
                new_best = validation["dice"] > best_val
                if new_best:
                    best_val, best_iteration = validation["dice"], iteration
                    torch.save(model.state_dict(), args.output_dir / "student_best.pth")
                with val_log.open("a", encoding="utf-8") as handle:
                    handle.write(
                        json.dumps(
                            {
                                "iteration": iteration,
                                **validation,
                                "is_new_best": new_best,
                                "safety_flags": safety_flags,
                                "safety_streak": safety_streak,
                                "x0_same_iteration_dice": x0_validation.get(iteration),
                            },
                            sort_keys=True,
                        ) + "\n"
                    )
                if safety_streak >= 5:
                    torch.save(
                        {
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "iteration": iteration,
                            "best_validation_dice": best_val,
                            "best_iteration": best_iteration,
                            "safety_flags": safety_flags,
                            "safety_streak": safety_streak,
                            "python_rng_state": random.getstate(),
                            "numpy_rng_state": np.random.get_state(),
                            "torch_rng_state": torch.get_rng_state(),
                            "cuda_rng_state_all": torch.cuda.get_rng_state_all(),
                        },
                        args.output_dir / "safety_stopped.pth",
                    )
                    (args.output_dir / "SAFETY_STOPPED").write_text(
                        json.dumps(
                            {
                                "iteration": iteration,
                                "flags": safety_flags,
                                "streak": safety_streak,
                            },
                            sort_keys=True,
                        )
                        + "\n",
                        encoding="utf-8",
                    )
                    raise RuntimeError(
                        f"Safety stop at {iteration}: {safety_flags}"
                    )
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "iteration": iteration,
                        "best_validation_dice": best_val,
                        "best_iteration": best_iteration,
                        "python_rng_state": random.getstate(),
                        "numpy_rng_state": np.random.get_state(),
                        "torch_rng_state": torch.get_rng_state(),
                        "cuda_rng_state_all": torch.cuda.get_rng_state_all(),
                    },
                    latest,
                )
            if iteration >= args.max_iterations:
                break

    final_path = args.output_dir / "student_final.pth"
    torch.save(model.state_dict(), final_path)
    final_validation = evaluate_validation(model)
    # Mandatory checkpoint load smoke: reconstruct and reproduce a validation forward.
    reloaded = SamUnet(args).cuda().eval()
    reloaded.load_state_dict(torch.load(final_path, map_location="cuda"))
    reloaded_validation = evaluate_validation(reloaded)
    if abs(reloaded_validation["dice"] - final_validation["dice"]) > 1e-12:
        raise RuntimeError("Reloaded Final checkpoint does not reproduce Validation")
    summary = {
        "experiment": args.experiment,
        "final_iteration": iteration,
        "final_checkpoint": str(final_path),
        "final_validation": final_validation,
        "reloaded_final_validation": reloaded_validation,
        "best_iteration_diagnostic_only": best_iteration,
        "best_validation_diagnostic_only": best_val,
        "test_evaluated": False,
        "nan_detected": False,
        "checkpoint_reproduced": True,
    }
    allocated_before_release = torch.cuda.memory_allocated()
    del reloaded, model, optimizer, train_loader, val_loader
    torch.cuda.empty_cache()
    summary["cuda_memory"] = {
        "allocated_before_release": allocated_before_release,
        "allocated_after_release": torch.cuda.memory_allocated(),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "TRAINING_COMPLETE").write_text("complete\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    parsed = parse_args()
    parsed.output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=str(parsed.output_dir / "train.log"),
        level=logging.INFO,
        format="[%(asctime)s] %(message)s",
    )
    main()
