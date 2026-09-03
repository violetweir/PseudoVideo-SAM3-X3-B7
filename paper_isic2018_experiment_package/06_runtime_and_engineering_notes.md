# Runtime And Engineering Notes

The slow training was traced mainly to CPU-side data loading/augmentation and resize overhead rather than GPU compute saturation. The later runs used an offline resized cache at 256 and more dataloader workers.

## Short Throughput/Debug Runs

| Experiment | Batch | Labeled | Workers | Final Iter | Val Dice | Test Dice | Note |
|---|---|---|---|---|---|---|---|
| s3_cache_b12_w8_400iter | 12 |  | 8 | 400 | 0.627730 | 0.639306 | short throughput/debug run with resized cache |
| s3_loader_gpu_b12_l6_w8_400iter | 12 | 6 | 8 | 400 | 0.620262 | 0.624723 | short loader/GPU benchmark |
| s3_loader_gpu_b12_l6_w12_400iter | 12 | 6 | 12 | 400 | 0.634986 | 0.647414 | short loader/GPU benchmark |
| s3_light_aug_heavy_b12_w8_400iter | 12 |  | 8 | 400 | 0.619931 | 0.626814 | short augmentation benchmark |

These are 400-iteration diagnostic runs. They should be cited only for implementation/runtime discussion, not as final segmentation accuracy results.

## Cache

The 256 resized cache contains 2075 train, 259 validation, and 260 test records. Cache path:

`work/isic18_round1_round2a_from_pseudovideo_full/isic_resized_cache_s256`
