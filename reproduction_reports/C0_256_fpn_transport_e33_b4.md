# C0_256 SAM3-FPN foreground transport with e33

> Date: 2026-08-26  
> Server: `violet@222.31.141.50`  
> Repository: `/Data_8TB/lht/PseudoVideo-SAM3-X3-B7`  
> Scope: fixed `b4`, validation/test only, 100 targets per split.

## Question

Test the LoRA epoch-33 merged SAM3 checkpoint on the SAM3-native
`sam3enc_fpn_foreground_transport` route family. Two factors are separated:

1. propagation teacher: C0 checkpoint versus SAM3-e33;
2. KNN topology: SAM3-base FPN descriptors versus SAM3-e33 FPN descriptors.

No target GT is used to extract features or construct routes. GT is consumed only
for the final Dice audit.

## Checkpoint and feature audit

SAM3-e33 checkpoint:

```text
work/rerun_c0_256_sam3knn_s256_base/medsam3_lora_b0_b6_e50/
  e33_full_evaluation/e33_merged_video.pt
SHA256: 2eb33d5be28baf8d579eddfec2a75d2ee7b8a0c26a600603983da2db99a52280
```

The e33 FPN cache has shape `(8,1000,1536)`. Against the SAM3-base FPN cache:

| Audit | Value |
|---|---:|
| mean row cosine | 0.595362 |
| median row cosine | 0.629107 |
| minimum row cosine | 0.052354 |
| mean L2 descriptor delta | 0.888325 |
| maximum absolute delta | 0.232330 |

The difference is large enough to rule out accidental reuse of SAM3-base features.

## Results

| FPN KNN topology | Propagation teacher | Validation b4 | Test b4 |
|---|---|---:|---:|
| SAM3-base | C0 `ft_1pct_merged_video.pt` | **0.878823** | **0.887058** |
| SAM3-base | SAM3-e33 | 0.860369 | 0.884635 |
| SAM3-e33 | SAM3-e33 | 0.850918 | 0.868344 |

Factor deltas:

| Controlled change | Validation | Test |
|---|---:|---:|
| C0 teacher -> e33 teacher, fixed base-FPN graph | -0.018454 | -0.002423 |
| base-FPN -> e33-FPN graph, fixed e33 teacher | -0.009451 | -0.016292 |

Paired target audit:

| Controlled change | Split | Win rate | Median delta | Bootstrap mean-delta 95% CI |
|---|---|---:|---:|---:|
| teacher | validation | 31% | -0.012784 | [-0.052250,+0.013439] |
| teacher | test | 23% | -0.013829 | [-0.032025,+0.030149] |
| topology | validation | 51% | +0.000207 | [-0.037354,+0.013937] |
| topology | test | 62% | +0.001321 | [-0.051394,+0.012568] |

The e33 topology wins slightly on many test targets but loses substantially on a
small number of targets, producing a negative mean delta. This is consistent with
the earlier Round-2B observation that e33 descriptors do not provide a better
long-bridge KNN topology even when e33 is a stronger propagation teacher.

## Conclusion

SAM3-e33 should not replace SAM3-base in the FPN foreground-transport KNN. It also
does not improve this fixed b4 route family as the propagation checkpoint. The
best tested FPN-transport configuration remains:

```text
SAM3-base FPN topology + C0 ft_1pct_merged_video.pt propagation
validation b4 = 0.878823
test b4       = 0.887058
```

For e33 itself, the existing base-KNN target-pooling route remains more suitable
(`test b4 = 0.904009`) than either FPN-transport combination tested here.

## Artifacts

- e33 feature root:
  `work/kvasir_1pct_anchors/stage1_feature_knn_sam3enc_fpn_transport_e33_s256/`
- base-FPN plus e33 teacher root:
  `work/kvasir_1pct_anchors/stage1_feature_knn_sam3enc_fpn_transport_base_b4_e33_teacher/`
- checkpoint-specific extractor:
  `scripts/extract_sam3_fpn_transport_checkpoint.py`
