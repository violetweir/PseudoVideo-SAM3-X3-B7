#!/usr/bin/env python3
from __future__ import annotations
import csv, json
from pathlib import Path

root = Path('/Data_8TB/lht/PseudoVideo-SAM3-X3-B7/work/kvasir_1pct_anchors')
models = ['base_no_ft','ft_1pct','ft_5pct','ft_10pct','ft_20pct_latest_after_crash']
rows=[]
for name in models:
    summary_path = root/'model_routes'/name/'summary.json'
    phase_path = root/'model_routes'/name/'test_pool0'/'selections.jsonl'
    summary = json.loads(summary_path.read_text())['test_pool0']
    selections = [json.loads(l) for l in phase_path.read_text().splitlines() if l.strip()]
    rows.append({
        'model': name,
        'targets': summary['n'],
        'test_dice_weighted': summary['dice_mean'],
        'test_dice_std': summary['dice_std'],
        'quality_mean': summary['quality_mean'],
        'direct_selected': summary['route_type_counts'].get('direct',0),
        'one_bridge_selected': summary['route_type_counts'].get('one_bridge',0),
        'two_bridges_selected': summary['route_type_counts'].get('two_bridges',0),
        'output_root': str(root/'model_routes'/name),
    })
summary_dir = root/'summaries'
summary_dir.mkdir(parents=True, exist_ok=True)
(summary_dir/'weighted_test_summary.json').write_text(json.dumps(rows, indent=2, sort_keys=True)+'\n')
with (summary_dir/'weighted_test_summary.csv').open('w', newline='') as f:
    writer=csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader(); writer.writerows(rows)
md=['| model | targets | test Dice weighted | quality mean | route counts (direct/one/two) |','|---|---:|---:|---:|---|']
for r in rows:
    md.append(f"| {r['model']} | {r['targets']} | {r['test_dice_weighted']:.6f} | {r['quality_mean']:.6f} | {r['direct_selected']}/{r['one_bridge_selected']}/{r['two_bridges_selected']} |")
(summary_dir/'weighted_test_summary.md').write_text('\n'.join(md)+'\n')
print('\n'.join(md))
