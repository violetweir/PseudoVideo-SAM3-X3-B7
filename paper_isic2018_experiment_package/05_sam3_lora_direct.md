# SAM3 LoRA And Direct Text-Only Results

## LoRA Selection

- Selected LoRA epoch for clean Round2A: epoch `27`.
- Epoch27 direct validation Dice from training log: `0.889317`.
- LoRA text/category prompt: `skin lesion`.
- Epoch27 merged checkpoint: `work/isic18_round1_round2a_from_pseudovideo_full/round2a_fixed_knn_lora_teacher/lora_epoch_27_merged_video.pt`.

## Direct Text-Only Tests

| Run | Split | Dice | IoU | Metric Resolution | Image Prompt | Route Propagation |
|---|---|---|---|---|---|---|
| epoch27 direct | test | 0.868218 | 0.792204 | 1008 | false | false |
| epoch27 direct | validation | 0.887161 | 0.818584 | 1008 | false | false |
| epoch50 direct | test | 0.874986 | 0.799037 | 1008 | false | false |
| epoch50 direct | test | 0.868009 | 0.789436 | 256 | false | false |

## Important Note

The direct setting here means no image prompt and no route propagation. It is category/text-only SAM3 inference with the class/category `skin lesion`.
