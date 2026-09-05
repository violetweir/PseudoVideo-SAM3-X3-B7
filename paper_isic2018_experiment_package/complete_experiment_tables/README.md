# Complete ISIC2018 Experiment Tables

This directory is the detailed ISIC2018 experiment ledger. It complements the concise paper-facing files one level above.

## Key Tables

- `tables/final_metric_ledger_long.csv`: curated metric ledger for main results, students, SAM3 direct tests, and B7 selector.
- `tables/manual_experiment_step_map.csv`: human-readable step map from protocol inheritance to baselines.
- `tables/manual_actual_or_equivalent_commands.csv`: actual or equivalent top-level commands used for the major runs.
- `tables/legacy_isic18_pseudovideo_full.csv`: explicit table for the old inherited ISIC run.
- `tables/commands_extracted_from_scripts.csv`: command templates extracted from shell/Python scripts. Variables such as `$SAMPY`, `$DATA`, `$ROUND1` are preserved.
- `tables/pipeline_log_step_timeline.csv`: timestamped step messages extracted from pipeline/nohup/launch logs.
- `tables/all_json_scalar_metrics_long.csv`: scalar values from every JSON artifact under ISIC-related experiment roots.
- `tables/all_jsonl_numeric_summaries.csv`: record counts and numeric summaries for JSONL artifacts.
- `tables/validation_curves_extracted.csv`: validation curves extracted from JSONL logs when available.
- `tables/train_curves_extracted.csv`: training curves extracted from JSONL logs when available.
- `tables/all_isic_artifacts.csv`: full inventory of ISIC-related artifacts.

## Resolution Warning

SAM3 direct text-only has both 1008 and 256 metric-resolution runs. For comparison against 256 student masks, use the `sam3_epoch50_direct_text_only_s256` row.
