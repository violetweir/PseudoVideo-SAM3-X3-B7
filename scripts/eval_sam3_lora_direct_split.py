#!/usr/bin/env python3
"""Evaluate a SAM3 LoRA checkpoint with the trainer's ordinary direct-Dice protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from sam3.model.model_misc import SAM3Output
from sam3.model_builder import build_sam3_image_model

from train_sam3_lora_kvasir_e50 import (
    COCOSegmentDataset,
    LoRAConfig,
    apply_lora_to_model,
    collate_fn_api,
    load_lora_weights,
)


def move_to_device(obj, device):
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, list):
        return [move_to_device(item, device) for item in obj]
    if isinstance(obj, tuple):
        return tuple(move_to_device(item, device) for item in obj)
    if isinstance(obj, dict):
        return {key: move_to_device(value, device) for key, value in obj.items()}
    if hasattr(obj, "__dataclass_fields__"):
        for field in obj.__dataclass_fields__:
            setattr(obj, field, move_to_device(getattr(obj, field), device))
    return obj


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--lora",
        type=Path,
        default=None,
        help="Optional LoRA weights. Omit to evaluate the untouched base model.",
    )
    parser.add_argument("--split", choices=("valid", "test"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--effective-resolution",
        type=int,
        default=1008,
        help=(
            "Information and metric resolution. RGB is downsampled to this size "
            "and then restored to SAM3's fixed 1008 input size."
        ),
    )
    parser.add_argument(
        "--prompt-mode",
        choices=("category", "empty"),
        default="category",
        help="Use the COCO category text or an empty text query.",
    )
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())
    base_checkpoint = Path(config["model"]["checkpoint_path"])
    data_dir = Path(config["training"]["data_dir"])
    lora_cfg = config["lora"]

    device = torch.device("cuda")
    model = build_sam3_image_model(
        device="cuda",
        compile=False,
        checkpoint_path=str(base_checkpoint),
        load_from_HF=False,
        bpe_path="/Data_8TB/lht/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz",
        eval_mode=False,
    )
    if args.lora is not None:
        model = apply_lora_to_model(
            model,
            LoRAConfig(
                rank=lora_cfg["rank"],
                alpha=lora_cfg["alpha"],
                dropout=lora_cfg["dropout"],
                target_modules=lora_cfg["target_modules"],
                apply_to_vision_encoder=lora_cfg["apply_to_vision_encoder"],
                apply_to_text_encoder=lora_cfg["apply_to_text_encoder"],
                apply_to_geometry_encoder=lora_cfg["apply_to_geometry_encoder"],
                apply_to_detr_encoder=lora_cfg["apply_to_detr_encoder"],
                apply_to_detr_decoder=lora_cfg["apply_to_detr_decoder"],
                apply_to_mask_decoder=lora_cfg["apply_to_mask_decoder"],
            ),
        )
        load_lora_weights(model, str(args.lora))
    model = model.to(device).eval()

    dataset = COCOSegmentDataset(data_dir=data_dir, split=args.split)
    model_input_resolution = dataset.resolution
    if args.effective_resolution > model_input_resolution:
        raise ValueError(
            f"effective resolution {args.effective_resolution} exceeds fixed model "
            f"input resolution {model_input_resolution}"
        )
    if args.prompt_mode == "empty":
        dataset.categories = {category_id: "" for category_id in dataset.categories}
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=lambda batch: collate_fn_api(
            batch, dict_key="input", with_seg_masks=True
        ),
        num_workers=args.num_workers,
        pin_memory=True,
    )

    dice_values = []
    iou_values = []
    nonempty_values = []
    valid_query_counts = []
    with torch.inference_mode():
        for batch_dict in tqdm(loader, desc=f"LoRA direct {args.split}"):
            input_batch = move_to_device(batch_dict["input"], device)
            if args.effective_resolution != model_input_resolution:
                low_resolution = F.interpolate(
                    input_batch.img_batch,
                    size=(args.effective_resolution, args.effective_resolution),
                    mode="bilinear",
                    align_corners=False,
                    antialias=True,
                )
                input_batch.img_batch = F.interpolate(
                    low_resolution,
                    size=(model_input_resolution, model_input_resolution),
                    mode="bilinear",
                    align_corners=False,
                    antialias=True,
                )
            with torch.autocast("cuda", dtype=torch.bfloat16):
                outputs_list = model(input_batch)
            targets = [model.back_convert(target) for target in input_batch.find_targets]
            with SAM3Output.iteration_mode(
                outputs_list, iter_mode=SAM3Output.IterMode.ALL_STEPS_PER_STAGE
            ) as outputs_iter:
                final_outputs = list(outputs_iter)[-1][-1]

            logits = final_outputs["pred_logits"].float().squeeze(-1)
            mask_probs = torch.sigmoid(final_outputs["pred_masks"].float())
            valid_queries = torch.sigmoid(logits[0]) >= 0.5
            if bool(valid_queries.any()):
                foreground_probability = mask_probs[0, valid_queries].amax(dim=0)
            else:
                foreground_probability = torch.zeros_like(mask_probs[0, 0])
            gt_model_resolution = targets[-1]["masks"].bool().any(dim=0)
            gt = F.interpolate(
                gt_model_resolution[None, None].float(),
                size=(args.effective_resolution, args.effective_resolution),
                mode="nearest",
            )[0, 0].bool()
            foreground_probability = F.interpolate(
                foreground_probability[None, None],
                size=gt.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )[0, 0]
            prediction = foreground_probability >= 0.5
            intersection = float((prediction & gt).sum())
            pred_area = float(prediction.sum())
            gt_area = float(gt.sum())
            union = pred_area + gt_area - intersection
            dice_values.append(
                1.0
                if pred_area + gt_area == 0
                else 2.0 * intersection / (pred_area + gt_area)
            )
            iou_values.append(1.0 if union == 0 else intersection / union)
            nonempty_values.append(float(prediction.any()))
            valid_query_counts.append(int(valid_queries.sum()))

    result = {
        "base_checkpoint": str(base_checkpoint.resolve()),
        "lora_checkpoint": (
            str(args.lora.resolve()) if args.lora is not None else None
        ),
        "split": args.split,
        "count": len(dice_values),
        "direct_dice": float(np.mean(dice_values)),
        "direct_dice_median": float(np.median(dice_values)),
        "direct_dice_quantiles": {
            str(q): float(np.quantile(dice_values, q))
            for q in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
        },
        "direct_iou": float(np.mean(iou_values)),
        "nonempty_rate": float(np.mean(nonempty_values)),
        "valid_query_count_mean": float(np.mean(valid_query_counts)),
        "protocol": {
            "class_probability_threshold": 0.5,
            "mask_probability_threshold": 0.5,
            "query_reduction": "foreground_union",
            "effective_source_resolution": args.effective_resolution,
            "model_input_resolution": model_input_resolution,
            "metric_resolution": args.effective_resolution,
            "checkpoint_training_resolution": config.get("hardware", {}).get(
                "training_resolution", 1008
            ),
            "prompt_mode": args.prompt_mode,
            "query_text": "colon polyp" if args.prompt_mode == "category" else "",
            "route_propagation": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
