# Round3 T3/T4：Constrained Pseudo-Temporal LoRA Adaptation

> Completed: 2026-08-28T12:39:39.859579+08:00  
> Root: `/Data_8TB/lht/PseudoVideo-SAM3-X3-B7/work/rerun_c0_256_round3_tracker_lora_t3_t4`

## Supervision audit

```json
{
  "num_supervised_positive_frames": 32789,
  "num_supervised_empty_frames": 0,
  "num_unsupervised_frames": 3891,
  "num_object_positive": 32789,
  "num_object_negative": 0,
  "num_object_ignore": 3891,
  "bug_found": false
}
```

Bridge frames without pseudo masks are IGNORE for both segmentation and object presence. Bug found: **False**.

## LoRA boundary

| Group | Trainable | Frozen | Total | Targets |
|---|---:|---:|---:|---:|
| T3 | 59,392 | 860,055,224 | 860,114,616 | 32 |
| T4 | 90,112 | 860,055,224 | 860,145,336 | 52 |

## Validation-best

| Model | b0 | b1 | b2 | b3 | b4 | b5 | b6 | Mean b3-b6 | b6-b0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| T0 | 0.737071 | 0.823352 | 0.849015 | 0.875069 | 0.868348 | 0.882818 | 0.884577 | 0.877703 | +0.147506 |
| T3 | 0.742675 | 0.823877 | 0.849319 | 0.876857 | 0.869155 | 0.881981 | 0.885157 | 0.878288 | +0.142482 |
| T4 | 0.765167 | 0.834099 | 0.852659 | 0.877558 | 0.871243 | 0.882961 | 0.885764 | 0.879382 | +0.120597 |

T3 best step: 1750; T4 best step: 2500.

## Validation gain over T0

```json
{
  "T3": {
    "delta_b0": 0.005604062084241246,
    "delta_b1": 0.0005242301139625205,
    "delta_b2": 0.0003039677269963681,
    "delta_b3": 0.0017885160096677843,
    "delta_b4": 0.0008068219448293945,
    "delta_b5": -0.0008375703207542662,
    "delta_b6": 0.0005803857462669537,
    "delta_mean_b3_b6": 0.0005845383450023833
  },
  "T4": {
    "delta_b0": 0.028095630410161387,
    "delta_b1": 0.010746595403202663,
    "delta_b2": 0.003643593829124825,
    "delta_b3": 0.0024889269799976965,
    "delta_b4": 0.002894859152327345,
    "delta_b5": 0.0001428318540638296,
    "delta_b6": 0.0011872503244136912,
    "delta_mean_b3_b6": 0.001678467077700585
  }
}
```

## Formal test

| Model | b0 | b1 | b2 | b3 | b4 | b5 | b6 | Mean b3-b6 | b6-b0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| T0 | 0.812347 | 0.825082 | 0.866180 | 0.885061 | 0.894624 | 0.895240 | 0.904065 | 0.894748 | +0.091719 |
| T3 | 0.817333 | 0.838983 | 0.872081 | 0.894604 | 0.894898 | 0.898076 | 0.905820 | 0.898350 | +0.088487 |
| T4 | 0.824417 | 0.845883 | 0.872679 | 0.895297 | 0.895415 | 0.898779 | 0.906295 | 0.898946 | +0.081878 |

## 20-target B6 path sensitivity

| Group | Normal | Shuffle | Normal-Shuffle |
|---|---:|---:|---:|
| T3 | 0.931039 | 0.931010 | +0.000029 |
| T4 | 0.931670 | 0.931672 | -0.000002 |

## Freeze integrity

- T3 changed frozen tensors: 0; nonzero frozen grads: 0.
- T4 changed frozen tensors: 0; nonzero frozen grads: 0.
- Initialization hash: `2eb33d5be28baf8d579eddfec2a75d2ee7b8a0c26a600603983da2db99a52280`.

## Actual inserted modules

### T3 (32 modules)

- `transformer.encoder.layers.0.cross_attn_image.k_proj`
- `transformer.encoder.layers.0.cross_attn_image.out_proj`
- `transformer.encoder.layers.0.cross_attn_image.q_proj`
- `transformer.encoder.layers.0.cross_attn_image.v_proj`
- `transformer.encoder.layers.0.self_attn.k_proj`
- `transformer.encoder.layers.0.self_attn.out_proj`
- `transformer.encoder.layers.0.self_attn.q_proj`
- `transformer.encoder.layers.0.self_attn.v_proj`
- `transformer.encoder.layers.1.cross_attn_image.k_proj`
- `transformer.encoder.layers.1.cross_attn_image.out_proj`
- `transformer.encoder.layers.1.cross_attn_image.q_proj`
- `transformer.encoder.layers.1.cross_attn_image.v_proj`
- `transformer.encoder.layers.1.self_attn.k_proj`
- `transformer.encoder.layers.1.self_attn.out_proj`
- `transformer.encoder.layers.1.self_attn.q_proj`
- `transformer.encoder.layers.1.self_attn.v_proj`
- `transformer.encoder.layers.2.cross_attn_image.k_proj`
- `transformer.encoder.layers.2.cross_attn_image.out_proj`
- `transformer.encoder.layers.2.cross_attn_image.q_proj`
- `transformer.encoder.layers.2.cross_attn_image.v_proj`
- `transformer.encoder.layers.2.self_attn.k_proj`
- `transformer.encoder.layers.2.self_attn.out_proj`
- `transformer.encoder.layers.2.self_attn.q_proj`
- `transformer.encoder.layers.2.self_attn.v_proj`
- `transformer.encoder.layers.3.cross_attn_image.k_proj`
- `transformer.encoder.layers.3.cross_attn_image.out_proj`
- `transformer.encoder.layers.3.cross_attn_image.q_proj`
- `transformer.encoder.layers.3.cross_attn_image.v_proj`
- `transformer.encoder.layers.3.self_attn.k_proj`
- `transformer.encoder.layers.3.self_attn.out_proj`
- `transformer.encoder.layers.3.self_attn.q_proj`
- `transformer.encoder.layers.3.self_attn.v_proj`
### T4 (52 modules)

- `sam_mask_decoder.transformer.final_attn_token_to_image.k_proj`
- `sam_mask_decoder.transformer.final_attn_token_to_image.out_proj`
- `sam_mask_decoder.transformer.final_attn_token_to_image.q_proj`
- `sam_mask_decoder.transformer.final_attn_token_to_image.v_proj`
- `sam_mask_decoder.transformer.layers.0.cross_attn_image_to_token.k_proj`
- `sam_mask_decoder.transformer.layers.0.cross_attn_image_to_token.out_proj`
- `sam_mask_decoder.transformer.layers.0.cross_attn_image_to_token.q_proj`
- `sam_mask_decoder.transformer.layers.0.cross_attn_image_to_token.v_proj`
- `sam_mask_decoder.transformer.layers.0.cross_attn_token_to_image.k_proj`
- `sam_mask_decoder.transformer.layers.0.cross_attn_token_to_image.out_proj`
- `sam_mask_decoder.transformer.layers.0.cross_attn_token_to_image.q_proj`
- `sam_mask_decoder.transformer.layers.0.cross_attn_token_to_image.v_proj`
- `sam_mask_decoder.transformer.layers.1.cross_attn_image_to_token.k_proj`
- `sam_mask_decoder.transformer.layers.1.cross_attn_image_to_token.out_proj`
- `sam_mask_decoder.transformer.layers.1.cross_attn_image_to_token.q_proj`
- `sam_mask_decoder.transformer.layers.1.cross_attn_image_to_token.v_proj`
- `sam_mask_decoder.transformer.layers.1.cross_attn_token_to_image.k_proj`
- `sam_mask_decoder.transformer.layers.1.cross_attn_token_to_image.out_proj`
- `sam_mask_decoder.transformer.layers.1.cross_attn_token_to_image.q_proj`
- `sam_mask_decoder.transformer.layers.1.cross_attn_token_to_image.v_proj`
- `transformer.encoder.layers.0.cross_attn_image.k_proj`
- `transformer.encoder.layers.0.cross_attn_image.out_proj`
- `transformer.encoder.layers.0.cross_attn_image.q_proj`
- `transformer.encoder.layers.0.cross_attn_image.v_proj`
- `transformer.encoder.layers.0.self_attn.k_proj`
- `transformer.encoder.layers.0.self_attn.out_proj`
- `transformer.encoder.layers.0.self_attn.q_proj`
- `transformer.encoder.layers.0.self_attn.v_proj`
- `transformer.encoder.layers.1.cross_attn_image.k_proj`
- `transformer.encoder.layers.1.cross_attn_image.out_proj`
- `transformer.encoder.layers.1.cross_attn_image.q_proj`
- `transformer.encoder.layers.1.cross_attn_image.v_proj`
- `transformer.encoder.layers.1.self_attn.k_proj`
- `transformer.encoder.layers.1.self_attn.out_proj`
- `transformer.encoder.layers.1.self_attn.q_proj`
- `transformer.encoder.layers.1.self_attn.v_proj`
- `transformer.encoder.layers.2.cross_attn_image.k_proj`
- `transformer.encoder.layers.2.cross_attn_image.out_proj`
- `transformer.encoder.layers.2.cross_attn_image.q_proj`
- `transformer.encoder.layers.2.cross_attn_image.v_proj`
- `transformer.encoder.layers.2.self_attn.k_proj`
- `transformer.encoder.layers.2.self_attn.out_proj`
- `transformer.encoder.layers.2.self_attn.q_proj`
- `transformer.encoder.layers.2.self_attn.v_proj`
- `transformer.encoder.layers.3.cross_attn_image.k_proj`
- `transformer.encoder.layers.3.cross_attn_image.out_proj`
- `transformer.encoder.layers.3.cross_attn_image.q_proj`
- `transformer.encoder.layers.3.cross_attn_image.v_proj`
- `transformer.encoder.layers.3.self_attn.k_proj`
- `transformer.encoder.layers.3.self_attn.out_proj`
- `transformer.encoder.layers.3.self_attn.q_proj`
- `transformer.encoder.layers.3.self_attn.v_proj`

## Conclusion

At least one constrained LoRA variant preserved path sensitivity and exceeded the frozen T0 long-chain validation score.

Did LoRA preserve the original SAM3 long-chain temporal prior? **Yes**.
