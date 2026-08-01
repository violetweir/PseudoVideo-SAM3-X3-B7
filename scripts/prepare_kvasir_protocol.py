#!/usr/bin/env python3
from __future__ import annotations
import json, random
from pathlib import Path

src = Path('/Data_8TB/lht/DG-GroupUNet/experiments/wacv2027/T02_fresh_polyp_hf_sources/raw_hf_snapshots/kvasir-seg/snapshot')
out = Path('/Data_8TB/lht/PseudoVideo-SAM3-X3-B7/work/kvasir_1pct_anchors/protocol')
out.mkdir(parents=True, exist_ok=True)
rows=[]
for split in ['train','validation','test']:
    for idx,line in enumerate((src/split/'metadata.jsonl').read_text().splitlines()):
        if not line.strip():
            continue
        r=json.loads(line)
        image=(src/split/r['file_name']).resolve()
        mask=(src/split/r['mask_file_name']).resolve()
        sid=Path(r['file_name']).stem
        rows.append({
            'height': None,
            'width': None,
            'image_path': str(image),
            'mask_path': str(mask),
            'file_name': str(image),
            'mask_file_name': str(mask),
            'merged_dataset': 'kvasir-seg',
            'merged_id': f'kvasir-seg::{sid}',
            'sample_id': sid,
            'source_dataset': 'kvasir-seg',
            'split': split,
        })
train=sorted([r for r in rows if r['split']=='train'], key=lambda r:r['merged_id'])
rng=random.Random(2026)
shuffled=train[:]
rng.shuffle(shuffled)
support=sorted(shuffled[:max(1, round(len(train)*0.01))], key=lambda r:r['merged_id'])
support=[{**r,'frozen_image_path':r['image_path'],'frozen_mask_path':r['mask_path']} for r in support]

def write_jsonl(path, arr):
    path.write_text(''.join(json.dumps(x, ensure_ascii=False, sort_keys=True)+'\n' for x in arr), encoding='utf-8')
write_jsonl(out/'merged_manifest.jsonl', sorted(rows, key=lambda r:(r['split'],r['merged_id'])))
write_jsonl(out/'support_manifest.jsonl', support)
(out/'frozen_labeled_images.txt').write_text('\n'.join(r['image_path'] for r in support)+'\n', encoding='utf-8')
(out/'protocol_summary.json').write_text(json.dumps({'data_root':str(src),'counts':{s:sum(r['split']==s for r in rows) for s in ['train','validation','test']},'support_count':len(support),'support_ids':[r['merged_id'] for r in support]}, indent=2, sort_keys=True)+'\n')
print(out)
print('support_count', len(support))
