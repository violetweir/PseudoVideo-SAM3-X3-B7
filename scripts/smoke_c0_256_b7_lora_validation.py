#!/usr/bin/env python3
"""Run one complete validation item through the patched e50 LoRA trainer."""

from __future__ import annotations

import torch

from train_sam3_lora_kvasir_e50 import (
    CORE_LOSS_KEY,
    COCOSegmentDataset,
    SAM3Output,
    SAM3TrainerNative,
    collate_fn_api,
)


CONFIG = "configs/c0_256_base_b7_medsam3_lora_e50.yaml"


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
    trainer = SAM3TrainerNative(CONFIG, multi_gpu=False)
    data_dir = trainer.config["training"]["data_dir"]
    dataset = COCOSegmentDataset(data_dir=data_dir, split="valid")
    batch = collate_fn_api([dataset[0]], dict_key="input", with_seg_masks=True)
    input_batch = move_to_device(batch["input"], trainer.device)

    trainer.model.eval()
    with torch.no_grad():
        with torch.autocast(
            device_type=trainer.device.type,
            dtype=torch.bfloat16,
            enabled=trainer.device.type == "cuda",
        ):
            outputs_list = trainer.model(input_batch)

        find_targets = [
            trainer._unwrapped_model.back_convert(target)
            for target in input_batch.find_targets
        ]
        for targets in find_targets:
            for key, value in targets.items():
                if isinstance(value, torch.Tensor):
                    targets[key] = value.to(trainer.device)

        with SAM3Output.iteration_mode(
            outputs_list, iter_mode=SAM3Output.IterMode.ALL_STEPS_PER_STAGE
        ) as outputs_iter:
            for stage_outputs, stage_targets in zip(outputs_iter, find_targets):
                for outputs in stage_outputs:
                    outputs["indices"] = trainer.matcher(outputs, stage_targets)
                    for aux_out in outputs.get("aux_outputs", []):
                        aux_out["indices"] = trainer.matcher(aux_out, stage_targets)

        loss_dict = trainer.loss_wrapper(outputs_list, find_targets)
        loss = loss_dict[CORE_LOSS_KEY]
    print(
        "VALIDATION_SMOKE_OK",
        f"loss={float(loss):.6f}",
        f"dtype={loss.dtype}",
        f"device={loss.device}",
    )


if __name__ == "__main__":
    main()
