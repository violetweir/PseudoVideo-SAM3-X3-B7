# ISIC2018 Paper Experiment Package

This folder is a paper-facing consolidation of the ISIC2018 experiments under:

`/Data_8TB/lht/PseudoVideo-SAM3-X3-B7/work/isic18_round1_round2a_from_pseudovideo_full`

Use this package as the first stop when writing the paper. The raw logs and JSON files remain in `work/`; this folder contains cleaned tables, notes, and source mapping.

## What Is Inside

- `00_executive_summary.md`: concise claims and numbers for Results.
- `01_experiment_protocol.md`: dataset, split, resolution, and prompt/protocol notes.
- `02_main_results.md`: paper-ready main comparisons.
- `03_round_progression_and_ablation.md`: S2/S3/X3/X4 progression and B7 selector notes.
- `04_bridge_analysis.md`: target-pooling and long-chain bridge analysis.
- `05_sam3_lora_direct.md`: LoRA/direct SAM3 results and the 256 vs 1008 distinction.
- `06_runtime_and_engineering_notes.md`: loader/cache/speed notes from the debugging runs.
- `tables/`: CSV tables for manuscript or spreadsheet import.
- `figures/`: copied bridge-benefit figure.
- `raw_records/`: copied master JSON and earlier records.
- `source_map.csv`: where every important number came from.

## Most Important Caution

The SAM3 direct text-only results have two metric resolutions. Do not mix them:

- `SAM3_epoch50 direct text-only @1008`: Dice `0.874986`
- `SAM3_epoch50 direct text-only @256`: Dice `0.868009`

For fair comparison with the 256-resolution student/bridge masks, use the `@256` value.
