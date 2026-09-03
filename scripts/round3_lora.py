#!/usr/bin/env python3
"""Minimal, auditable LoRA support for the Round3 SAM3 tracker experiments."""
from __future__ import annotations

import math
import re
from typing import Any

import torch
from torch import nn


MEMORY_ATTENTION_PATTERN = re.compile(
    r"^transformer\.encoder\.layers\.\d+\."
    r"(?:self_attn|cross_attn_image)\."
    r"(?:q_proj|k_proj|v_proj|out_proj)$"
)
MASK_DECODER_CROSS_ATTENTION_PATTERN = re.compile(
    r"^sam_mask_decoder\.transformer\."
    r"(?:layers\.\d+\.cross_attn_(?:image_to_token|token_to_image)|"
    r"final_attn_token_to_image)\."
    r"(?:q_proj|k_proj|v_proj|out_proj)$"
)


class LoRALinear(nn.Module):
    """Frozen Linear layer plus a trainable low-rank residual."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float):
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")
        self.base = base
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.dropout = nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity()
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        factory = {"device": base.weight.device, "dtype": base.weight.dtype}
        self.lora_A = nn.Parameter(torch.empty(self.rank, base.in_features, **factory))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, self.rank, **factory))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = torch.nn.functional.linear(self.dropout(inputs), self.lora_A)
        residual = torch.nn.functional.linear(residual, self.lora_B)
        return self.base(inputs) + residual * self.scaling

    def effective_update_norm(self) -> torch.Tensor:
        return (self.lora_B.float() @ self.lora_A.float()).norm() * self.scaling


def target_kind(name: str) -> str | None:
    if MEMORY_ATTENTION_PATTERN.fullmatch(name):
        return "memory_attention"
    if MASK_DECODER_CROSS_ATTENTION_PATTERN.fullmatch(name):
        return "mask_decoder_cross_attention"
    return None


def target_names(tracker: nn.Module, policy: str) -> list[str]:
    allowed = {"memory_attention"}
    if policy == "memory_attention_decoder_lora":
        allowed.add("mask_decoder_cross_attention")
    elif policy != "memory_attention_lora":
        raise ValueError(f"Unsupported LoRA policy: {policy}")
    selected = []
    for name, module in tracker.named_modules():
        if target_kind(name) in allowed:
            if not isinstance(module, nn.Linear):
                raise TypeError(f"LoRA target is not nn.Linear: {name} ({type(module).__name__})")
            selected.append(name)
    return sorted(selected)


def _replace_submodule(root: nn.Module, name: str, replacement: nn.Module) -> None:
    parent_name, child_name = name.rsplit(".", 1)
    parent = root.get_submodule(parent_name)
    parent.add_module(child_name, replacement)


def inject_lora(
    tracker: nn.Module,
    policy: str,
    rank: int,
    alpha: float,
    dropout: float,
) -> dict[str, Any]:
    selected = target_names(tracker, policy)
    expected = 32 if policy == "memory_attention_lora" else 52
    if len(selected) != expected:
        raise RuntimeError(f"Expected {expected} LoRA targets for {policy}, found {len(selected)}: {selected}")
    kinds: dict[str, int] = {}
    for name in selected:
        module = tracker.get_submodule(name)
        kind = target_kind(name)
        kinds[kind] = kinds.get(kind, 0) + 1
        _replace_submodule(tracker, name, LoRALinear(module, rank, alpha, dropout))
    return {
        "policy": policy,
        "rank": int(rank),
        "alpha": float(alpha),
        "dropout": float(dropout),
        "target_count": len(selected),
        "target_kind_counts": kinds,
        "target_modules": selected,
    }


def lora_update_norms(tracker: nn.Module) -> dict[str, float]:
    parameters = dict(tracker.named_parameters())
    per_module = {}
    for name, left in parameters.items():
        if not name.endswith(".lora_A"):
            continue
        prefix = name.removesuffix(".lora_A")
        right = parameters[f"{prefix}.lora_B"]
        module = tracker.get_submodule(prefix)
        scaling = float(getattr(module, "scaling"))
        per_module[prefix] = float(((right.float() @ left.float()).norm() * scaling).detach().cpu())
    if not per_module:
        raise RuntimeError("No LoRA parameter pairs found while computing update norm")
    total = math.sqrt(sum(value * value for value in per_module.values()))
    return {
        "total": total,
        "maximum": max(per_module.values(), default=0.0),
        "mean": sum(per_module.values()) / max(len(per_module), 1),
    }
