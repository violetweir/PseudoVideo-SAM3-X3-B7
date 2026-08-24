#!/usr/bin/env python3
"""Evaluate untouched base SAM3 with the LoRA trainer's direct-Val-Dice protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from sam3.model.model_misc import SAM3Output
from sam3.model_builder import build_sam3_image_model

from train_sam3_lora_kvasir_e50 import COCOSegmentDataset, collate_fn_api


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
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    device = torch.device("cuda")
    model = build_sam3_image_model(
        device="cuda",
        compile=False,
        checkpoint_path=str(args.checkpoint),
        load_from_HF=False,
        bpe_path="/Data_8TB/lht/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz",
        eval_mode=False,
    ).to(device).eval()
    dataset = COCOSegmentDataset(data_dir=args.data_dir, split="valid")
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=lambda batch: collate_fn_api(
            batch, dict_key="input", with_seg_masks=True
        ),
        num_workers=2,
        pin_memory=True,
    )

    dice_values = []
    iou_values = []
    nonempty_values = []
    valid_query_counts = []
    with torch.inference_mode():
        for batch_dict in tqdm(loader, desc="Base SAM3 direct validation"):
            input_batch = move_to_device(batch_dict["input"], device)
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
            gt = targets[-1]["masks"].bool().any(dim=0)
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
                1.0 if pred_area + gt_area == 0 else 2.0 * intersection / (pred_area + gt_area)
            )
            iou_values.append(1.0 if union == 0 else intersection / union)
            nonempty_values.append(float(prediction.any()))
            valid_query_counts.append(int(valid_queries.sum()))

    result = {
        "checkpoint": str(args.checkpoint.resolve()),
        "split": "validation",
        "count": len(dice_values),
        "direct_dice": float(np.mean(dice_values)),
        "direct_dice_median": float(np.median(dice_values)),
        "direct_iou": float(np.mean(iou_values)),
        "nonempty_rate": float(np.mean(nonempty_values)),
        "valid_query_count_mean": float(np.mean(valid_query_counts)),
        "protocol": {
            "class_probability_threshold": 0.5,
            "mask_probability_threshold": 0.5,
            "query_reduction": "foreground_union",
            "training_resolution": 1008,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
