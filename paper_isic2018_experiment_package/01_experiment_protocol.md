# Experiment Protocol

## Dataset

| Item | Value |
|---|---|
| dataset | ISIC2018 |
| train_images | 2075 |
| validation_images | 259 |
| test_images | 260 |
| frozen_labeled_images | 21 |
| protocol_root | work/isic18_pseudovideo_full/protocol |
| data_root | work/isic18_1pct_protocol/data |
| resize_cache | work/isic18_round1_round2a_from_pseudovideo_full/isic_resized_cache_s256 |

## Core Setting

All current ISIC experiments inherited the old `isic18_pseudovideo_full` protocol. The frozen labeled set contains 21 labeled images, while training/validation/test contain 2075/259/260 images respectively.

The student and bridge experiments use 256-resolution masks. The SAM3 direct evaluator can run with a 1008 model input while reporting metrics either at 1008 or at 256; both are recorded separately in this package.

## Prompt Setting

For SAM3 LoRA/direct tests, the actual COCO category/text prompt is `skin lesion`. Some evaluator JSON files contain a stale hard-coded `query_text` field; the loaded category in the run is still `skin lesion`.
