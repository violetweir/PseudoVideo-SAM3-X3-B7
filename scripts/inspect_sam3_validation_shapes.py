#!/usr/bin/env python3
"""Inspect one validation batch for direct semantic Dice integration."""

import torch
import torch.nn.functional as F

from train_sam3_lora_kvasir_e50 import (
    COCOSegmentDataset,
    SAM3Output,
    SAM3TrainerNative,
    collate_fn_api,
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


def main():
    trainer = SAM3TrainerNative(
        "configs/c0_256_base_b7_medsam3_lora_e50.yaml", multi_gpu=False
    )
    dataset = COCOSegmentDataset(
        data_dir=trainer.config["training"]["data_dir"], split="valid"
    )
    batch = collate_fn_api([dataset[0]], dict_key="input", with_seg_masks=True)
    input_batch = move_to_device(batch["input"], trainer.device)
    targets = [trainer._unwrapped_model.back_convert(t) for t in input_batch.find_targets]
    print("TARGETS", len(targets))
    for index, target in enumerate(targets):
        print("TARGET", index, {k: tuple(v.shape) if isinstance(v, torch.Tensor) else type(v).__name__ for k, v in target.items()})
    trainer.model.eval()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        outputs = trainer.model(input_batch)
    with SAM3Output.iteration_mode(outputs, iter_mode=SAM3Output.IterMode.ALL_STEPS_PER_STAGE) as iterator:
        final = list(iterator)[-1][-1]
    print("OUTPUT", {k: tuple(v.shape) if isinstance(v, torch.Tensor) else type(v).__name__ for k, v in final.items() if k in ("pred_logits", "pred_masks", "pred_boxes")})
    logits = final["pred_logits"].float().squeeze(-1)
    mask_probs = torch.sigmoid(final["pred_masks"].float())
    valid = torch.sigmoid(logits[0]) >= 0.5
    foreground = (
        mask_probs[0, valid].amax(dim=0)
        if bool(valid.any())
        else torch.zeros_like(mask_probs[0, 0])
    )
    gt = targets[-1]["masks"].bool().any(dim=0)
    foreground = F.interpolate(
        foreground[None, None], size=gt.shape[-2:], mode="bilinear", align_corners=False
    )[0, 0]
    prediction = foreground >= 0.5
    intersection = (prediction & gt).sum(dtype=torch.float64)
    denominator = prediction.sum(dtype=torch.float64) + gt.sum(dtype=torch.float64)
    dice = torch.ones((), device=gt.device, dtype=torch.float64) if float(denominator) == 0.0 else 2.0 * intersection / denominator
    print("DIRECT_DICE_SMOKE_OK", float(dice), "valid_queries", int(valid.sum()))


if __name__ == "__main__":
    main()
