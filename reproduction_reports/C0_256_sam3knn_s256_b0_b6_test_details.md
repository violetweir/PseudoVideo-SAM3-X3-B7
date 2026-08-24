# C0-256 SAM3-KNN@256 b0-b6 current-route test details

Date: 2026-08-24 (Asia/Shanghai)

## Scope and protocol

This report evaluates the current route pipeline before applying the ongoing
SAM3 LoRA training checkpoints.

- SAM3 propagation checkpoint: untouched base `sam3.pt`
- KNN feature extractor: frozen SAM3 image trunk
- KNN input resolution: 256 x 256
- propagation canvas: 256 x 256
- route modes: target pooling and patch correspondence
- candidate steps: direct/b0 through b6
- test targets: 100
- candidates per target: 14 (seven steps x two modes)
- route results: 1,400, with no missing predictions
- selector: frozen X3 val-best (`0.826032` at iteration 28,800) + B7
- checkpoint and protocol selection did not use test GT
- this test execution was explicitly authorized by the user

`b7` is a self-consistency/model-agreement score. It is not Dice. All Dice
values below are computed against test GT only after routes and checkpoints
were frozen.

## Main results

| result | Test Dice | Test IoU / oracle | gap to oracle |
|---|---:|---:|---:|
| X3 val-best direct prediction | 0.853920 | 0.771766 IoU | - |
| **X3 val-best + B7, b0-b6** | **0.895432** | **0.921471 oracle** | **0.026039** |
| X3 final direct prediction | 0.858591 | 0.778503 IoU | - |
| X3 final + B7, b0-b6 (control) | 0.897684 | 0.921471 oracle | 0.023786 |

The official mainline remains X3 val-best because it was selected on
validation. X3 final is reported only as a test audit/control even though its
test result is slightly higher.

Against the original C0-256-base report:

| selector | old C0-256-base B7 | current B7 | delta |
|---|---:|---:|---:|
| X3 val-best | 0.886130 | 0.895432 | +0.009302 |
| X3 final | 0.875302 | 0.897684 | +0.022382 |

## Per-step test results

Each mode contains exactly 100 results at every step. `combined` averages all
200 results from the two modes at that step; it is not B7 selection.

| step | target Dice | target median | target cycle | patch Dice | patch median | patch cycle | combined Dice | combined median | combined cycle |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| direct / b0 | 0.798144 | 0.946492 | 0.856412 | 0.706645 | 0.935329 | 0.752706 | 0.752395 | 0.940924 | 0.804559 |
| b1 | 0.846623 | 0.946073 | 0.868034 | 0.754897 | 0.941709 | 0.742701 | 0.800760 | 0.943610 | 0.805367 |
| b2 | 0.870032 | 0.946816 | 0.899518 | 0.788470 | 0.942226 | 0.642437 | 0.829251 | 0.944863 | 0.770978 |
| b3 | 0.862253 | 0.952737 | 0.916853 | 0.859413 | 0.943405 | 0.637434 | 0.860833 | 0.946810 | 0.777143 |
| b4 | 0.873930 | 0.948239 | 0.924861 | 0.857625 | 0.944580 | 0.627593 | **0.865778** | 0.945365 | 0.776227 |
| b5 | 0.869552 | 0.949360 | 0.939041 | 0.855339 | 0.941804 | 0.619069 | 0.862446 | 0.945890 | 0.779055 |
| b6 | **0.874004** | 0.948556 | 0.937421 | 0.853581 | **0.946291** | 0.628343 | 0.863793 | **0.947342** | 0.782882 |

Observations:

- Target pooling is consistently stronger than patch correspondence.
- Target pooling peaks at b6 (`0.874004`), narrowly above b4 (`0.873930`).
- Patch correspondence peaks at b3 (`0.859413`).
- The combined per-step mean peaks at b4 (`0.865778`).
- B7 improves over every fixed step by selecting a different candidate for
  each target; mainline B7 reaches `0.895432`.

## X3 test breakdown

| checkpoint | overall | small | medium | large | nonempty rate |
|---|---:|---:|---:|---:|---:|
| X3 val-best | 0.853920 | 0.853231 | 0.899043 | 0.836993 | 1.00 |
| X3 final control | 0.858591 | 0.849371 | 0.909392 | 0.843217 | 1.00 |

The size groups contain 24 small, 21 medium, and 55 large test images.

## B7-selected bridge distribution

### X3 val-best mainline

| selected step | targets | mean selected-mask Dice | mean B7 |
|---|---:|---:|---:|
| direct / b0 | 6 | 0.923986 | 0.890429 |
| b1 | 10 | 0.910104 | 0.945749 |
| b2 | 13 | 0.951636 | 0.960118 |
| b3 | 16 | 0.896660 | 0.846731 |
| b4 | 17 | 0.800416 | 0.872425 |
| b5 | 21 | 0.931198 | 0.900237 |
| b6 | 17 | 0.883420 | 0.908643 |

Mode distribution:

| selected mode | targets | mean selected-mask Dice | mean B7 |
|---|---:|---:|---:|
| target pooling | 68 | 0.905496 | 0.894583 |
| patch correspondence | 32 | 0.874046 | 0.911900 |

The patch mode has a higher mean B7 but lower true Dice. This is an important
remaining calibration gap: self-consistency is occasionally overconfident on
patch candidates.

### X3 final control

| selected step | targets | mean selected-mask Dice | mean B7 |
|---|---:|---:|---:|
| direct / b0 | 6 | 0.915381 | 0.880881 |
| b1 | 8 | 0.939025 | 0.963674 |
| b2 | 14 | 0.952140 | 0.963432 |
| b3 | 16 | 0.887681 | 0.835339 |
| b4 | 19 | 0.812934 | 0.874485 |
| b5 | 23 | 0.939236 | 0.905417 |
| b6 | 14 | 0.870207 | 0.906064 |

Mode counts are 69 target-pooling and 31 patch-correspondence selections.

## B7 score distribution

| checkpoint | mean | minimum | p10 | p25 | median | p75 | p90 | maximum |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| X3 val-best | 0.900124 | 0.293458 | 0.734254 | 0.868449 | 0.956526 | 0.977178 | 0.985281 | 0.990862 |
| X3 final | 0.899729 | 0.286489 | 0.729970 | 0.871462 | 0.963550 | 0.977719 | 0.984001 | 0.991046 |

These values describe selector confidence only; they must not be reported as
segmentation performance.

## Artifacts

Server root:

```text
work/rerun_c0_256_sam3knn_s256_base/current_base_test/
```

Important files:

```text
bridge_b0_b6_test_metrics.json
bridge_b0_b6_test_metrics.tsv
x3_test_results.json
x3_best_b7_b0_b6_test.jsonl
x3_best_b7_b0_b6_test.summary.json
x3_best_b7_report/b7_report.json
x3_final_b7_b0_b6_test.jsonl
x3_final_b7_b0_b6_test.summary.json
x3_final_b7_report/b7_report.json
run.log
COMPLETE
```
