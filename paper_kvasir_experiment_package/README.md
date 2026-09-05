# Kvasir Experiment Package

This folder consolidates Kvasir-related experiments from:

- `work/kvasir_1pct_anchors`
- `work/clinicdb_external_kvasir8`

It mirrors the ISIC organization, but Kvasir has more exploratory router, prompt, LoRA, and external-evaluation branches.

## Dataset Snapshot

- train: `800`
- validation: `100`
- test: `100`

## Main Tables

- `tables/curated_metric_ledger_long.csv`: curated ledger covering dataset counts, SynFoC, C3/router summaries, known finetune/repro summaries, ClinicDB external setup, and all route-family Dice rows.
- `tables/route_family_and_bridge_dice.csv`: direct/bridge Dice extracted from every discovered route-family or propagation summary JSON.
- `tables/top_aggregate_route_family_dice.csv`: top aggregate route-family Dice rows, sorted descending. Start here for paper tables.
- `tables/top_route_family_dice.csv`: top route-family Dice rows including per-target selector details, useful for auditing but not directly paper-ready.
- `tables/all_json_scalar_metrics_long.csv`: all scalar metrics from every JSON.
- `tables/all_jsonl_numeric_summaries.csv`: JSONL record counts and numeric summaries.
- `tables/log_metric_lines.csv`: metric-looking lines extracted from logs.
- `tables/commands_extracted.csv`: command lines/templates extracted from `.sh`, `.py`, `.log`, and `.md`.
- `tables/manual_experiment_step_and_command_map.csv`: human-readable command/source map.
- `tables/all_kvasir_artifacts.csv`: complete artifact inventory.

## Curated Highlights

| Group | Experiment | Split | Metric | Value |
|---|---|---|---|---|
| dataset | kvasir_1pct_anchors | test | count | 100 |
| dataset | kvasir_1pct_anchors | train | count | 800 |
| dataset | kvasir_1pct_anchors | validation | count | 100 |
| baseline | SynFoC-Kvasir-SAM | test | dice | 0.761503 |
| baseline | SynFoC-Kvasir-UNet | test | dice | 0.768113 |
| baseline | SynFoC-Kvasir-SAM | validation | best_sam_dice | 0.731031 |
| router | c3_B_direct | test | per_target_mean | 0.832448 |
| router | c3_B_direct | test | oracle | 0.913114 |
| router | c3_B_direct | test | sel_medoid | 0.871230 |
| router | c3_C_top1 | test | per_target_mean | 0.862525 |
| router | c3_C_top1 | test | oracle | 0.929697 |
| router | c3_C_top1 | test | sel_medoid | 0.895218 |
| router | c3_D_all | test | per_target_mean | 0.869648 |
| router | c3_D_all | test | oracle | 0.931232 |
| router | c3_D_all | test | sel_medoid | 0.894539 |
| router | c3_A_within | test | per_target_mean | 0.894659 |
| router | c3_A_within | test | oracle | 0.931232 |
| router | c3_A_within | test | sel_medoid | 0.894539 |
| main_or_legacy_kvasir | ft_1pct_repro_20260806 | test | max_bridge_5.dice_mean | 0.898055 |
| main_or_legacy_kvasir | ft_1pct_repro_20260806 | test | max_bridge_6.dice_mean | 0.902643 |
| main_or_legacy_kvasir | ft_1pct_repro_20260806 | test_pool0 | dice_mean | 0.886850 |
| main_or_legacy_kvasir | ft_1pct_plus_hq_pseudo_lr025_eval1 | test | max_bridge_5.dice_mean | 0.864411 |
| main_or_legacy_kvasir | ft_1pct_plus_hq_pseudo_lr025_eval1 | test | max_bridge_6.dice_mean | 0.860351 |
| main_or_legacy_kvasir | ft_1pct_plus_hq_pseudo_lr025_eval1 | test_pool0 | dice_mean | 0.836675 |
| clinicdb_external | clinicdb_external_kvasir8 |  | support_count | 8 |
| clinicdb_external | clinicdb_external_kvasir8 |  | target_count | 61 |
| clinicdb_external | clinicdb_external_kvasir8 |  | bridge_pool_count | 800 |

## Top Aggregate Route/Bridge Dice Rows

| Root | Family/Bridge | Dice |
|---|---|---|
| work/kvasir_1pct_anchors/stage1_feature_knn_sam3enc_s1008_lesion_knn/sam3enc_lesion__gt_lesion/eval_base_no_ft_b7_forward | bridge_5 | 0.919017 |
| work/kvasir_1pct_anchors/stage1_feature_knn_sam3enc_s1008/sam3enc_anchor_conditioned_target_pooling__knn_cond/eval_lora_p491_e20 | bridge_4 | 0.915389 |
| work/kvasir_1pct_anchors/stage1_feature_knn_sam3enc_s1008/sam3enc_anchor_conditioned_target_pooling__knn_cond/eval_lora_p491_e20 | bridge_1 | 0.915374 |
| work/kvasir_1pct_anchors/stage1_feature_knn_sam3enc_s1008/sam3enc_anchor_conditioned_patch_correspondence/eval_lora_p491_e20 | bridge_6 | 0.914955 |
| work/kvasir_1pct_anchors/stage1_feature_knn_sam3enc_s1008/sam3enc_anchor_conditioned_patch_correspondence/eval_lora_p491_e20 | bridge_5 | 0.914931 |
| work/kvasir_1pct_anchors/stage1_feature_knn_sam3enc_s1008/sam3enc_anchor_conditioned_target_pooling__knn_cond/eval_lora_p491_e20 | bridge_5 | 0.913622 |
| work/kvasir_1pct_anchors/stage1_feature_knn_sam3enc_s1008/sam3enc_anchor_conditioned_patch_correspondence__knn_pooled/eval_lora_p491_e20 | bridge_2 | 0.913541 |
| work/kvasir_1pct_anchors/stage1_feature_knn_sam3enc_s1008/sam3enc_anchor_conditioned_target_pooling__knn_cond/eval_lora_p491_e20 | bridge_6 | 0.913106 |
| work/kvasir_1pct_anchors/stage1_feature_knn_sam3enc_s1008/sam3enc_anchor_conditioned_target_pooling__knn_cond/eval_lora_p491_e20 | bridge_3 | 0.913090 |
| work/kvasir_1pct_anchors/stage1_feature_knn_sam3enc_s1008/sam3enc_anchor_conditioned_patch_correspondence/eval_lora_p491_e20 | bridge_1 | 0.912923 |
| work/kvasir_1pct_anchors/lora_experiment/test_eval/lora_p491_e20/anchor_conditioned_patch_correspondence/eval_base_no_ft_b7_forward | bridge_6 | 0.911171 |
| work/kvasir_1pct_anchors/stage1_feature_knn_sam3enc_imr_s256/sam3enc_imr__tb050/eval_lora_p491_e20 | bridge_6 | 0.910984 |
| work/kvasir_1pct_anchors/stage1_feature_knn_sam3enc_s1008/sam3enc_anchor_conditioned_target_pooling/eval_lora_p491_e20 | bridge_1 | 0.910832 |
| work/kvasir_1pct_anchors/stage1_feature_knn_sam3enc_s1008/sam3enc_anchor_conditioned_patch_correspondence/eval_lora_p491_e20 | bridge_2 | 0.909835 |
| work/kvasir_1pct_anchors/stage1_feature_knn_sam3enc_s256/sam3enc_anchor_conditioned_patch_correspondence__knn_cond/eval_lora_p491_e20 | bridge_4 | 0.908213 |
| work/kvasir_1pct_anchors/stage1_feature_knn_sam3enc_s1008/sam3enc_anchor_conditioned_target_pooling/eval_lora_p491_e20 | bridge_5 | 0.908170 |
| work/kvasir_1pct_anchors/stage1_feature_knn_sam3enc_imr_s256/sam3enc_imr__tb050/eval_lora_p491_e20 | bridge_4 | 0.907447 |
| work/kvasir_1pct_anchors/stage1_feature_knn_sam3enc_s1008/sam3enc_anchor_conditioned_patch_correspondence__knn_pooled/eval_lora_p491_e20 | bridge_5 | 0.906607 |
| work/kvasir_1pct_anchors/stage1_feature_knn_sam3enc_imr_s256/sam3enc_pyramid_contrast/eval_lora_p491_e20 | bridge_3 | 0.906124 |
| work/kvasir_1pct_anchors/stage1_feature_knn_sam3enc_s1008/sam3enc_anchor_conditioned_patch_correspondence__knn_pooled/eval_lora_p491_e20 | bridge_4 | 0.906097 |

## Notes

The package is intentionally exhaustive. For paper tables, start from `curated_metric_ledger_long.csv` and `top_route_family_dice.csv`; for auditing/reproduction, use the all-JSON/all-JSONL/commands/log tables.
