# ISIC2018 Experiment Records

Created: 2026-09-03 16:37 CST

Remote project:

`/Data_8TB/lht/PseudoVideo-SAM3-X3-B7`

Target experiment root:

`work/isic18_round1_round2a_from_pseudovideo_full`

This record summarizes the ISIC2018 experiments that reused the old `isic18_pseudovideo_full` dataset/protocol setting.

## Dataset And Protocol

| Item | Value |
|---|---:|
| Train images | 2075 |
| Validation images | 259 |
| Test images | 260 |
| Frozen labeled images | 21 |
| Protocol | `work/isic18_pseudovideo_full/protocol` |
| Labeled list | `work/isic18_pseudovideo_full/protocol/frozen_labeled_images.txt` |
| Data root used by scripts | `work/isic18_1pct_protocol/data` |
| Resize cache | `work/isic18_round1_round2a_from_pseudovideo_full/isic_resized_cache_s256` |

## Main Test Dice

| Method | Test Dice | Notes |
|---|---:|---|
| SynFoC-SAM | 0.873360 | `work/isic18_1pct_protocol/synfoc/summary.json` |
| SAM3_epoch50 direct text-only @1008 | 0.874986 | No image prompt, no route propagation; category prompt is `skin lesion`; metric resolution 1008 |
| Ours X4_best | 0.870414 | Recomputed from exported `X4_best` test masks at 256x256 |
| SAM3_epoch50 direct text-only @256 | 0.868009 | No image prompt, no route propagation; category prompt is `skin lesion`; metric resolution 256 |
| SAM3_ep27 direct text-only | 0.868218 | No image prompt, no route propagation; category prompt is `skin lesion` |
| Ours X3_best + epoch27 B7 selector | 0.861104 | `selected_gt_dice_evaluation_only`; selector evaluation, not student-only mask Dice |
| Ours X4_best + epoch27 B7 selector | 0.860339 | `selected_gt_dice_evaluation_only`; selector evaluation, not student-only mask Dice |
| SCSAM-SAM | 0.852200 | Rerun on 2026-09-03, 260 test images |
| Ours S3 best | 0.854872 | From Round1 student training summary |
| Ours S2 best | 0.848341 | From Round1 student training summary |
| SynFoC-UNet | 0.841385 | `work/isic18_1pct_protocol/synfoc/summary.json` |
| SCSAM-UNet | 0.830954 | Rerun on 2026-09-03, 260 test images |

## Student Networks

| Student | Best/Final | Validation Dice | Test Dice | Source |
|---|---|---:|---:|---|
| S2 | best @ 12000 | 0.850589 | 0.848341 | `round1_sam3knn_s256_base/students/S2/summary.json` |
| S3 | best @ 26600 | 0.852072 | 0.854872 | `round1_sam3knn_s256_base/students/S3/summary.json` |
| X3 | diagnostic best @ 22400 | 0.869182 | 0.859377 | Test Dice recomputed earlier from exported `X3_best` masks |
| X3 | final @ 40000 | 0.864878 | 0.864459 | Test Dice recomputed earlier from exported `X3_final` masks |
| X4 | diagnostic best @ 24000 | 0.870790 | 0.870414 | Test Dice recomputed from exported `X4_best` masks |
| X4 | final @ 40000 | 0.867375 | not evaluated | `round2a_fixed_knn_lora_teacher/students/X4/summary.json` |

## SAM3 / B7 Records

### SAM3_ep27 Direct Text-Only Test

Path:

`work/isic18_round1_round2a_from_pseudovideo_full/round1_sam3knn_s256_base/medsam3_lora_b0_b6_e50/direct_test_epoch27_category_merged.json`

| Metric | Value |
|---|---:|
| Direct Dice | 0.868218 |
| Direct IoU | 0.792204 |
| Count | 260 |
| Nonempty rate | 0.996154 |
| Valid query count mean | 1.146154 |

This uses the merged epoch27 SAM3 checkpoint with category/text prompting only, no image prompt and no route propagation. The evaluation script's JSON has a stale hard-coded `query_text` label, but the loaded COCO category is `skin lesion`.

### SAM3_epoch50 Direct Text-Only Test, 1008 Metric

Path:

`work/isic18_round1_round2a_from_pseudovideo_full/round1_sam3knn_s256_base/medsam3_lora_b0_b6_e50/direct_test_epoch50_category_merged.json`

| Metric | Value |
|---|---:|
| Direct Dice | 0.874986 |
| Direct IoU | 0.799037 |
| Count | 260 |
| Nonempty rate | 1.000000 |
| Valid query count mean | 1.007692 |

This uses the merged epoch50 SAM3 checkpoint with category/text prompting only, no image prompt and no route propagation. The evaluation script's JSON has a stale hard-coded `query_text` label, but the loaded COCO category is `skin lesion`.

### SAM3_epoch50 Direct Text-Only Test, 256 Metric

Path:

`work/isic18_round1_round2a_from_pseudovideo_full/round1_sam3knn_s256_base/medsam3_lora_b0_b6_e50/direct_test_epoch50_category_merged_s256.json`

| Metric | Value |
|---|---:|
| Direct Dice | 0.868009 |
| Direct IoU | 0.789436 |
| Count | 260 |
| Nonempty rate | 1.000000 |
| Valid query count mean | 1.103846 |

This uses the merged epoch50 SAM3 checkpoint with category/text prompting only, no image prompt and no route propagation. For this run, `effective_source_resolution=256`, `metric_resolution=256`, and `model_input_resolution=1008`.

### Base SAM3-KNN Target Pooling Test, b0-b6

| Bridge | Dice |
|---|---:|
| b0 | 0.715039 |
| b1 | 0.710054 |
| b2 | 0.716470 |
| b3 | 0.722435 |
| b4 | 0.724580 |
| b5 | 0.727414 |
| b6 | 0.728152 |

Base combined best b5: `0.708182`.

### Epoch27 LoRA Long-Chain Test, b0-b7

Path:

`work/isic18_round1_round2a_from_pseudovideo_full/round2a_epoch27_longchain_test_b0_b7/test_bridge_b0_b7.tsv`

| Scope | b0 | b1 | b2 | b3 | b4 | b5 | b6 | b7 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| combined | 0.850546 | 0.851250 | 0.858358 | 0.851794 | 0.853416 | 0.857424 | 0.857055 | 0.003139 |
| target_pooling | 0.852890 | 0.845234 | 0.856095 | 0.857645 | 0.856142 | 0.858849 | 0.858307 | 0.003307 |
| patch_correspondence | 0.848201 | 0.857266 | 0.860622 | 0.845943 | 0.850689 | 0.855998 | 0.855804 | 0.002971 |

The b7 long-chain value is abnormal and was not debugged further per instruction.

### B7 Selector Summaries

| Setting | Selected GT Dice | Oracle GT Dice | Coverage |
|---|---:|---:|---:|
| Round1 base X3_best B7 | 0.761549 | 0.788188 | 260/260 |
| Round2A epoch27 X3_best B7 | 0.861104 | 0.878812 | 260/260 |
| Round2A epoch27 X4_best B7 | 0.860339 | 0.878812 | 260/260 |

## LoRA

| Item | Value |
|---|---|
| Best LoRA epoch used for clean Round2A | epoch 27 |
| Epoch27 direct Val Dice | 0.889317 |
| LoRA weights | `round1_sam3knn_s256_base/medsam3_lora_b0_b6_e50/lora_weights/epoch_27_lora_weights.pt` |
| Merged checkpoint | `round2a_fixed_knn_lora_teacher/lora_epoch_27_merged_video.pt` |
| Text/category prompt | `skin lesion` |

## Clean Round2A Rerun

The clean rerun was started after moving the contaminated output directory aside:

`round2a_fixed_knn_lora_teacher_contaminated_20260901_200933`

Clean output directory:

`work/isic18_round1_round2a_from_pseudovideo_full/round2a_fixed_knn_lora_teacher`

Completion marker:

`work/isic18_round1_round2a_from_pseudovideo_full/ROUND2A_COMPLETE`

Completion time:

`2026-09-03 01:45:25 CST`

## Bridge Benefit Analysis

Path:

`work/isic18_round1_round2a_from_pseudovideo_full/analysis/target_pooling_bridge_benefit_b0_b6`

| Setting | Mean best-vs-b0 Delta | Median best-vs-b0 Delta | Delta >= 0.01 | Delta >= 0.03 | Delta >= 0.05 | Delta < 0 |
|---|---:|---:|---:|---:|---:|---:|
| Base | 0.048924 | 0.007678 | 115/260 | 61/260 | 42/260 | 64/260 |
| LoRA epoch27 | 0.017001 | 0.001870 | 58/260 | 29/260 | 16/260 | 71/260 |

Interpretation: after LoRA adaptation, bridge gains remain real but are concentrated in fewer hard cases; the main gain is from SAM3 domain adaptation rather than long-chain bridging.

## Logs And Outputs

| Artifact | Path |
|---|---|
| Main pipeline log | `work/isic18_round1_round2a_from_pseudovideo_full/pipeline.log` |
| Clean epoch27 Round2A log | `work/isic18_round1_round2a_from_pseudovideo_full/pipeline_round2a_epoch27_gpu1.log` |
| SCSAM rerun test log | `work/isic18_1pct_protocol/logs/scsam_test_best_rerun_20260903_160852.log` |
| SynFoC summary | `work/isic18_1pct_protocol/synfoc/summary.json` |
| Round2A X4 predictions | `work/isic18_round1_round2a_from_pseudovideo_full/round2a_fixed_knn_lora_teacher/predictions/X4_best` |
