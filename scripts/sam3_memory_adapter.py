#!/usr/bin/env python3
"""Zero-residual memory-read adapter for SAM3 tracker training and inference."""

from __future__ import annotations

import types

import torch
from torch import nn


class MemoryReadAdapter(nn.Module):
    """Channel MLP applied after frozen memory attention and before mask decoding."""

    def __init__(self, dim: int = 256, reduction: int = 4) -> None:
        super().__init__()
        hidden = dim // reduction
        self.norm = nn.LayerNorm(dim)
        self.down = nn.Linear(dim, hidden)
        self.activation = nn.GELU()
        self.up = nn.Linear(hidden, dim)
        # Function-preserving and trainable: zero residual at step 0 while the
        # final layer still receives gradients. Initializing both up=0 and
        # gamma=0 would create a dead branch.
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)
        self.gamma = nn.Parameter(torch.ones(()))
        self.enabled = True
        self.last_residual: torch.Tensor | None = None

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        channels_last = features.permute(0, 2, 3, 1)
        residual = self.up(self.activation(self.down(self.norm(channels_last))))
        residual = self.gamma * residual.permute(0, 3, 1, 2)
        self.last_residual = residual
        if not self.enabled:
            return features
        return features + residual

    def residual(self, features: torch.Tensor) -> torch.Tensor:
        return self.forward(features) - features


def attach_memory_read_adapter(
    tracker: nn.Module, reduction: int = 4
) -> MemoryReadAdapter:
    """Attach exactly one adapter and patch the memory-read output."""
    if hasattr(tracker, "t23_memory_read_adapter"):
        return tracker.t23_memory_read_adapter
    adapter = MemoryReadAdapter(dim=tracker.hidden_dim, reduction=reduction)
    tracker.add_module("t23_memory_read_adapter", adapter)
    original = tracker._prepare_memory_conditioned_features
    object.__setattr__(tracker, "_t23_original_prepare_memory", original)

    def wrapped(self, *args, **kwargs):
        features = self._t23_original_prepare_memory(*args, **kwargs)
        is_init = bool(kwargs.get("is_init_cond_frame", False))
        if is_init:
            return features
        return self.t23_memory_read_adapter(features)

    object.__setattr__(
        tracker,
        "_prepare_memory_conditioned_features",
        types.MethodType(wrapped, tracker),
    )
    return adapter


def load_memory_adapter(tracker: nn.Module, payload: dict) -> MemoryReadAdapter:
    config = payload.get("memory_adapter_config", {})
    adapter = attach_memory_read_adapter(
        tracker, reduction=int(config.get("reduction", 4))
    )
    adapter.load_state_dict(payload["memory_adapter_state_dict"], strict=True)
    return adapter
