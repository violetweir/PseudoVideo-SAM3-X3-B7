# Main Results

## Paper-Ready Main Table

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

## Baseline Notes

- SynFoC-SAM test Dice: `0.873360`.
- SCSAM-SAM test Dice: `0.852200` from the 2026-09-03 rerun.
- SynFoC-UNet and SCSAM-UNet are included as non-SAM branch references.

## Comparability Notes

Use `SAM3_epoch50 direct text-only @256 = 0.868009` when comparing to 256-resolution student masks. Keep `SAM3_epoch50 direct text-only @1008 = 0.874986` as a separate high-resolution direct-teacher result.
