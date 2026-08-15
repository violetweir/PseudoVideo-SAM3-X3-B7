#!/usr/bin/env python3
"""Merge trained SAM3 LoRA weights into the base sam3.pt video checkpoint.

The result is a full video checkpoint (base tracker + detector with the LoRA
delta folded into the detector weights) so the standard propagation-quality
eval pipeline can consume it.

Usage:
  python scripts/merge_sam3_lora_video_checkpoint.py \
    --base-checkpoint <sam3.pt> \
    --lora-weights <epoch_N_lora_weights.pt> \
    --rank 16 --alpha 32 --dropout 0.1 \
    --output <merged_video.pt>
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch
from sam3.model_builder import build_sam3_image_model


ROOT = Path("/Data_8TB/lht/PseudoVideo-SAM3-X3-B7")
LORA_LAYERS_PATH = ROOT / "MedSAM3" / "lora_layers.py"
spec = importlib.util.spec_from_file_location("lora_layers", LORA_LAYERS_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot import {LORA_LAYERS_PATH}")
lora_layers = importlib.util.module_from_spec(spec)
sys.modules["lora_layers"] = lora_layers
spec.loader.exec_module(lora_layers)


TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "out_proj",
    "qkv", "proj", "fc1", "fc2",
    "c_fc", "c_proj", "linear1", "linear2",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--lora-weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=32.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    args = parser.parse_args()

    # Build image model with LoRA applied (same structure as training).
    model = build_sam3_image_model(
        device="cpu",
        compile=False,
        checkpoint_path=str(args.base_checkpoint.resolve()),
        load_from_HF=False,
        bpe_path="/Data_8TB/lht/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz",
        eval_mode=False,
    )
    config = lora_layers.LoRAConfig(
        rank=args.rank,
        alpha=args.alpha,
        dropout=args.dropout,
        target_modules=list(TARGET_MODULES),
        apply_to_vision_encoder=True,
        apply_to_text_encoder=True,
        apply_to_geometry_encoder=True,
        apply_to_detr_encoder=True,
        apply_to_detr_decoder=True,
        apply_to_mask_decoder=True,
    )
    model = lora_layers.apply_lora_to_model(model, config)
    lora_layers.load_lora_weights(model, str(args.lora_weights))

    # Fold LoRA deltas into the base video checkpoint (detector.* namespace).
    # Standalone LoRALinear -> add delta to the matching ".weight" key.
    # q/k/v projections inside a replaced MultiheadAttentionLoRA -> add delta to
    # the fused "in_proj_weight" key at the corresponding slice.
    base = torch.load(args.base_checkpoint, map_location="cpu", weights_only=False)
    mha_parents = {}
    for name, module in model.named_modules():
        if isinstance(module, lora_layers.MultiheadAttentionLoRA):
            mha_parents[name] = module

    updated = []
    skipped = []
    for name, module in model.named_modules():
        if not isinstance(module, lora_layers.LoRALinear):
            continue
        if module.lora is None:
            continue
        delta = (
            module.lora.lora_A.data @ module.lora.lora_B.data
        ).T * module.lora.scaling  # (out, in)
        *parent_parts, attr = name.split(".")
        parent_name = ".".join(parent_parts)
        parent = mha_parents.get(parent_name)

        if parent is not None and attr in ("q_proj", "k_proj", "v_proj"):
            embed_dim = parent.embed_dim
            slices = {"q_proj": (0, embed_dim), "k_proj": (embed_dim, 2 * embed_dim), "v_proj": (2 * embed_dim, 3 * embed_dim)}
            start, end = slices[attr]
            dst = f"detector.{parent_name}.in_proj_weight"
            if dst not in base:
                skipped.append(dst)
                continue
            base[dst][start:end, :] += delta.to(base[dst].dtype)
            updated.append(dst)
        else:
            dst = f"detector.{name}.weight"
            if dst not in base:
                skipped.append(dst)
                continue
            base[dst] = base[dst] + delta.to(base[dst].dtype)
            updated.append(dst)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(base, args.output)
    summary = {
        "base": str(args.base_checkpoint),
        "lora": str(args.lora_weights),
        "output": str(args.output),
        "updated": len(updated),
        "skipped": skipped,
        "total_detector_keys": sum(1 for k in base if k.startswith("detector.")),
    }
    summary_path = args.output.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
