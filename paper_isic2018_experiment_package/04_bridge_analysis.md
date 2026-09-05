# Bridge Analysis

## Base SAM3-KNN Test, b0-b6

| Mode | b0 | b1 | b2 | b3 | b4 | b5 | b6 |
|---|---:|---:|---:|---:|---:|---:|---:|
| target_pooling | 0.715039 | 0.710054 | 0.716470 | 0.722435 | 0.724580 | 0.727414 | 0.728152 |
| patch_correspondence | 0.681840 | 0.690171 | 0.689701 | 0.690101 | 0.685299 | 0.688950 | 0.674552 |
| combined | 0.698439 | 0.700112 | 0.703086 | 0.706268 | 0.704940 | 0.708182 | 0.701352 |

The Patch Correspondence row is now included explicitly. The combined row averages both modes as in `base_test_bridge_b0_b6.tsv`.

## Epoch27 LoRA Long-Chain Test, b0-b7

| Scope | b0 | b1 | b2 | b3 | b4 | b5 | b6 | b7 |
|---|---|---|---|---|---|---|---|---|
| combined | 0.850546 | 0.851250 | 0.858358 | 0.851794 | 0.853416 | 0.857424 | 0.857055 | 0.003139 |
| target_pooling | 0.852890 | 0.845234 | 0.856095 | 0.857645 | 0.856142 | 0.858849 | 0.858307 | 0.003307 |
| patch_correspondence | 0.848201 | 0.857266 | 0.860622 | 0.845943 | 0.850689 | 0.855998 | 0.855804 | 0.002971 |

The b7 values in this long-chain run are abnormal and were not debugged further, following the instruction to focus on the first bridge-analysis item.

## Per-Target Bridge Benefit

| Setting | Mean Δ best-vs-b0 | Median Δ | Δ >= 0.01 | Δ >= 0.03 | Δ >= 0.05 | Δ < 0 |
|---|---|---|---|---|---|---|
| base | 0.048924 | 0.007678 | 115/260 | 61/260 | 42/260 | 64/260 |
| lora_epoch27 | 0.017001 | 0.001870 | 58/260 | 29/260 | 16/260 | 71/260 |

Interpretation: after LoRA adaptation, bridge gains remain real but are smaller and concentrated in fewer hard samples. This supports the claim that SAM3 domain adaptation is the dominant gain source on ISIC2018.

Figure: `figures/isic_target_pooling_bridge_benefit_distribution.png`
