#!/usr/bin/env python3
"""Small LoRA wrappers for SAM3 RouteCo experiments."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class LoRAConfig:
    rank: int = 4
    alpha: float = 8.0
    dropout: float = 0.0


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, config: LoRAConfig) -> None:
        super().__init__()
        self.base = base
        self.rank = config.rank
        self.scaling = config.alpha / max(1, config.rank)
        self.dropout = nn.Dropout(config.dropout) if config.dropout else nn.Identity()
        self.lora_a = nn.Linear(base.in_features, config.rank, bias=False)
        self.lora_b = nn.Linear(config.rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=5**0.5)
        nn.init.zeros_(self.lora_b.weight)
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.base(x)
        with torch.autocast(device_type=x.device.type, enabled=False):
            residual = self.dropout(x).to(self.lora_a.weight.dtype)
            residual = self.lora_b(self.lora_a(residual))
        residual = residual.to(base.dtype)
        return base + residual * self.scaling


class LoRAConv2d1x1(nn.Module):
    def __init__(self, base: nn.Conv2d, config: LoRAConfig) -> None:
        super().__init__()
        if base.kernel_size != (1, 1):
            raise ValueError("LoRAConv2d1x1 only supports 1x1 convolutions")
        self.base = base
        self.rank = config.rank
        self.scaling = config.alpha / max(1, config.rank)
        self.dropout = nn.Dropout2d(config.dropout) if config.dropout else nn.Identity()
        self.lora_a = nn.Conv2d(base.in_channels, config.rank, kernel_size=1, bias=False)
        self.lora_b = nn.Conv2d(config.rank, base.out_channels, kernel_size=1, bias=False)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=5**0.5)
        nn.init.zeros_(self.lora_b.weight)
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.base(x)
        with torch.autocast(device_type=x.device.type, enabled=False):
            residual = self.dropout(x).to(self.lora_a.weight.dtype)
            residual = self.lora_b(self.lora_a(residual))
        residual = residual.to(base.dtype)
        return base + residual * self.scaling


def _replace_child(parent: nn.Module, child_name: str, replacement: nn.Module) -> None:
    setattr(parent, child_name, replacement)


def attach_mask_decoder_lora(
    tracker: nn.Module,
    *,
    rank: int = 4,
    alpha: float = 8.0,
    dropout: float = 0.0,
    include_conv1x1: bool = True,
) -> list[str]:
    """Wrap Linear and optional 1x1 Conv2d modules under sam_mask_decoder."""
    if not hasattr(tracker, "sam_mask_decoder"):
        raise RuntimeError("Tracker has no sam_mask_decoder module")
    if hasattr(tracker, "routeco_lora_attached"):
        return list(tracker.routeco_lora_module_names)

    config = LoRAConfig(rank=rank, alpha=alpha, dropout=dropout)
    wrapped: list[str] = []
    for module_name, module in list(tracker.sam_mask_decoder.named_modules()):
        for child_name, child in list(module.named_children()):
            full_name = f"sam_mask_decoder.{module_name}.{child_name}" if module_name else f"sam_mask_decoder.{child_name}"
            if isinstance(child, nn.Linear):
                replacement = LoRALinear(child, config).to(child.weight.device)
                _replace_child(module, child_name, replacement)
                wrapped.append(full_name)
            elif (
                include_conv1x1
                and isinstance(child, nn.Conv2d)
                and child.kernel_size == (1, 1)
            ):
                replacement = LoRAConv2d1x1(child, config).to(child.weight.device)
                _replace_child(module, child_name, replacement)
                wrapped.append(full_name)

    if not wrapped:
        raise RuntimeError("No mask decoder Linear/1x1 Conv2d modules were wrapped")
    tracker.routeco_lora_attached = True
    tracker.routeco_lora_module_names = wrapped
    return wrapped


def lora_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu()
        for name, value in module.state_dict().items()
        if ".lora_a." in name or ".lora_b." in name
    }


def load_lora_state_dict(module: nn.Module, state: dict[str, torch.Tensor]) -> None:
    current = module.state_dict()
    missing = sorted(set(state) - set(current))
    if missing:
        raise RuntimeError(f"LoRA checkpoint has unknown keys: {missing[:8]}")
    current.update(state)
    module.load_state_dict(current, strict=True)
