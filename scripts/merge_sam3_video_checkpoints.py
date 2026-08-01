#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path
import torch

base_path = Path('/Data_8TB/lht/models/modelscope/models/facebook--sam3/snapshots/master/sam3.pt')
source_root = Path('/Data_8TB/lht/DG-GroupUNet/experiments/wacv2027/T11_sam3_lowlabel_ft_kvasir_budgets')
out_root = Path('/Data_8TB/lht/PseudoVideo-SAM3-X3-B7/work/kvasir_1pct_anchors/video_checkpoints')
out_root.mkdir(parents=True, exist_ok=True)
models = {
    'ft_1pct': source_root/'finetune_1pct_seed2026/checkpoints/checkpoint.pt',
    'ft_5pct': source_root/'finetune_5pct_seed2026/checkpoints/checkpoint.pt',
    'ft_10pct': source_root/'finetune_10pct_seed2026/checkpoints/checkpoint.pt',
    'ft_20pct_latest_after_crash': source_root/'finetune_20pct_seed2026/checkpoints/checkpoint.pt',
}
base = torch.load(base_path, map_location='cpu')
if not isinstance(base, dict) or 'detector.backbone.vision_backbone.trunk.pos_embed' not in base:
    raise RuntimeError('Unexpected base checkpoint format')
summary = []
for name, src in models.items():
    ckpt = torch.load(src, map_location='cpu')
    detector = ckpt.get('model') if isinstance(ckpt, dict) else None
    if detector is None:
        raise RuntimeError(f'{src} has no model key')
    merged = dict(base)
    updated = 0
    skipped = []
    for key, value in detector.items():
        dst = f'detector.{key}'
        if dst not in merged:
            skipped.append(key)
            continue
        if tuple(merged[dst].shape) != tuple(value.shape):
            raise RuntimeError(f'shape mismatch {dst}: {tuple(merged[dst].shape)} vs {tuple(value.shape)}')
        merged[dst] = value
        updated += 1
    out = out_root/f'{name}_merged_video.pt'
    torch.save(merged, out)
    summary.append({'name': name, 'source': str(src), 'output': str(out), 'updated_detector_keys': updated, 'skipped': skipped[:20], 'skipped_count': len(skipped), 'epoch': ckpt.get('epoch') if isinstance(ckpt, dict) else None, 'steps': ckpt.get('steps') if isinstance(ckpt, dict) else None})
(out_root/'merge_summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True)+'\n')
print(json.dumps(summary, indent=2, sort_keys=True))
