# Executive Summary

## Main Takeaways

1. On ISIC2018, the largest gain comes from SAM3 domain adaptation rather than long-chain bridging.
2. The best 256-resolution student result is `X4_best`, with test Dice `0.870414`.
3. Direct SAM3 LoRA with text/category-only prompting is very strong. The epoch50 checkpoint reaches Dice `0.874986` at 1008 metric resolution, but the fair 256-resolution value is `0.868009`.
4. Bridge gains remain present after LoRA, but they shrink and concentrate on fewer hard samples: mean best-vs-b0 gain falls from `0.048924` before LoRA to `0.017001` after LoRA.
5. B7 selector numbers are evaluation/selection Dice, not final standalone student-mask Dice.

## Main Test Dice Table

| Method | Test Dice | Metric Resolution | Setting |
|---|---|---|---|
| SynFoC-SAM | 0.873360 | reported_by_baseline | SAM baseline |
| SAM3_epoch50 direct text-only | 0.874986 | 1008 | text/category only: skin lesion; no image prompt |
| Ours X4_best student mask | 0.870414 | 256 | Round1+Round2A, student mask |
| SAM3_epoch50 direct text-only | 0.868009 | 256 | text/category only: skin lesion; no image prompt |
| SAM3_epoch27 direct text-only | 0.868218 | 1008 | text/category only: skin lesion; no image prompt |
| Ours X3_best + epoch27 B7 selector | 0.861104 | 256 | B7 selector evaluation |
| Ours X4_best + epoch27 B7 selector | 0.860339 | 256 | B7 selector evaluation |
| SCSAM-SAM | 0.852200 | reported_by_baseline | SAM baseline |
| Ours S3 best | 0.854872 | 256 | Round1 student |
| Ours S2 best | 0.848341 | 256 | Round1 student |
| SynFoC-UNet | 0.841385 | reported_by_baseline | UNet baseline |
| SCSAM-UNet | 0.830954 | reported_by_baseline | UNet branch |

## Suggested Paper Wording

For the 256-resolution comparison, report `X4_best` as the main student result and compare it against `SAM3_epoch50 direct text-only @256`, SCSAM, and SynFoC with explicit metric-resolution notes. The high-resolution `SAM3_epoch50 @1008` result can be used to discuss the upper performance of the adapted SAM3 teacher, but it should be labeled separately.
