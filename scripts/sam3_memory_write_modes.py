#!/usr/bin/env python3
"""Training-time SAM3 memory writing with explicit soft/hard/STE/GT modes."""

from __future__ import annotations

from typing import Any

import torch


MEMORY_WRITE_MODES = ("soft", "hard_detach", "hard_ste", "gt_teacher")


def choose_memory_mask(
    mask_logits: torch.Tensor,
    mode: str,
    gt_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    if mode not in MEMORY_WRITE_MODES:
        raise ValueError(f"Unknown memory-write mode: {mode}")
    probability = mask_logits.float().sigmoid()
    hard = (probability > 0.5).float()
    if mode == "soft":
        memory_mask = probability
    elif mode == "hard_detach":
        memory_mask = hard.detach()
    elif mode == "hard_ste":
        memory_mask = hard - probability.detach() + probability
    else:
        if gt_mask is None:
            raise ValueError("gt_teacher requires a GT mask for every frame")
        memory_mask = gt_mask.float()
        if memory_mask.ndim == 3:
            memory_mask = memory_mask.unsqueeze(0)
    stats = {
        "soft_mean": float(probability.detach().mean()),
        "hard_area": float(hard.detach().mean()),
        "memory_area": float(memory_mask.detach().mean()),
    }
    return memory_mask, stats


def encode_memory_with_mask(
    tracker: torch.nn.Module,
    *,
    image: torch.Tensor,
    current_vision_feats: list[torch.Tensor],
    feat_sizes: list[tuple[int, int]],
    memory_mask: torch.Tensor,
    object_score_logits: torch.Tensor,
) -> tuple[torch.Tensor, list[torch.Tensor], dict[str, float]]:
    batch = current_vision_feats[-1].size(1)
    channels = tracker.hidden_dim
    height, width = feat_sizes[-1]
    pixel_features = (
        current_vision_feats[-1]
        .permute(1, 2, 0)
        .view(batch, channels, height, width)
    )
    output = tracker.maskmem_backbone(
        pixel_features, memory_mask, skip_mask_sigmoid=True
    )
    features = tracker._maybe_clone(output["vision_features"])
    positions = [
        tracker._maybe_clone(item) for item in output["vision_pos_enc"]
    ]
    is_object = (object_score_logits > 0).float()
    features = features + (
        (1 - is_object[..., None, None])
        * tracker.no_obj_embed_spatial[..., None, None].expand(*features.shape)
    )
    stats = {
        "memory_feature_norm": float(
            features.detach().float().flatten(1).norm(dim=1).mean()
        ),
    }
    return features, positions, stats


def replace_output_memory(
    tracker: torch.nn.Module,
    output: dict[str, Any],
    *,
    image: torch.Tensor,
    current_vision_feats: list[torch.Tensor],
    feat_sizes: list[tuple[int, int]],
    mode: str,
    gt_mask: torch.Tensor | None,
) -> dict[str, float]:
    memory_mask, stats = choose_memory_mask(
        output["pred_masks_high_res"], mode, gt_mask
    )
    features, positions, feature_stats = encode_memory_with_mask(
        tracker,
        image=image,
        current_vision_feats=current_vision_feats,
        feat_sizes=feat_sizes,
        memory_mask=memory_mask,
        object_score_logits=output["object_score_logits"],
    )
    output["maskmem_features"] = features
    output["maskmem_pos_enc"] = positions
    stats.update(feature_stats)
    stats["object_pointer_norm"] = float(
        output["obj_ptr"].detach().float().norm(dim=-1).mean()
    )
    stats["objectness"] = float(
        output["object_score_logits"].detach().float().sigmoid().mean()
    )
    stats["sam_score"] = float(output["iou_score"].detach().float().mean())
    return stats
