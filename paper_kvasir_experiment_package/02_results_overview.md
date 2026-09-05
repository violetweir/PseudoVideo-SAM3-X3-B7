# Kvasir Results Overview

This overview is generated from discovered artifacts. The exhaustive source of truth is the CSV table set under `tables/`.

## Top Aggregate Route/Bridge Results

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

## Baseline And Router Highlights

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
