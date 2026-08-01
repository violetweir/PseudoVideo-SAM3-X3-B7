# Method Ladder

This project has two entry points:

- `scripts/run_pipeline.py`: the clean final reproduction path for
  `S27 X3 Final+B7`.
- `scripts/run_method_ladder.py`: the broader experimental ladder that shows
  how the method develops from simple SAM3 baselines to the final system.

## Ladder Stages

| Stage | Purpose | Main Script |
|---|---|---|
| B00 | SAM3 predicts each image independently with a generic category-free prompt. | `run_b00_sam3_single_image.py` |
| T19 | SC-SAM 16GT low-label student baseline on the fixed merged protocol. | `run_t19_scsam_baseline.py` |
| T20 | SynFoC 16GT low-label student baseline on the fixed merged protocol. | `run_t20_synfoc_baseline.py` |
| T17 | Single-image candidate pools and fixed 16 train GT support protocol. | `prepare_t17_autonomous_1pct.py`, `extract_t17_sam3_candidates.py` |
| T18 | Two-frame pseudo-video: retrieve one similar support image and propagate support visual memory to the query. | `prepare_t18_full_retrieval.py`, `eval_t18_pseudovideo_pilot.py` |
| E1 | Multi-step propagation study with depths K=1..5 and star/chain/hybrid memory structures. | `prepare_e1_multistep.py`, `eval_e1_multistep.py` |
| T21 | Frozen three-route pseudo-video protocol: direct, one-bridge, two-bridge. | `run_t21_dynamic_pseudovideo.py` |
| T22/T23 | SAM3 adaptation diagnostics using only 16 GT and pseudo-video supervision. These are diagnostic branches because memory adaptation can collapse. | `train_t22_sam3_tracker.py`, `train_t23_single_image_decoder.py` |
| T24 | Single-image student committee trained from 16 GT plus pseudo labels. | `run_t24_student.py` |
| S27 | Progressive student-audited pseudo-label expansion and X3 student training. | `run_s27_student.py` |
| B7 | Student-assisted multi-route selector over frozen SAM3 routes. | `run_t25_offline_analysis.py` |

## Why These Stages Matter

B00 answers: how far does SAM3 get without support images or pseudo-video memory?

T18 answers: can one annotated support image guide a different query image through
SAM3 video memory?

T19/T20 answer: how strong are ordinary low-label students when trained on the
same 16 support images, before adding SAM3 pseudo labels?

E1 answers: does increasing pseudo-video depth help or hurt, and are star,
chain, or hybrid structures more stable?

T21 answers: if one route is brittle, can three frozen route hypotheses give a
high oracle ceiling?

T22/T23 answer: can SAM3 itself be adapted from the 16 GT anchors and pseudo
labels? The observed collapse risk is why the final mainline freezes SAM3 route
generation and trains independent students instead.

T24/S27 answer: can a single-image student audit and improve frozen SAM3 route
selection without becoming a propagation anchor?

## Running The Ladder

Preview all commands:

```bash
cd /Data_8TB/lht/PseudoVideo-SAM3-X3-B7
/home/violet/anaconda3/envs/mkunet_mamba/bin/python scripts/run_method_ladder.py \
  --config configs/reproduction.toml \
  --dry-run
```

Run only the early baselines:

```bash
/home/violet/anaconda3/envs/mkunet_mamba/bin/python scripts/run_method_ladder.py \
  --config configs/reproduction.toml \
  --to-stage e1_hybrid
```

Run the final mainline:

```bash
/home/violet/anaconda3/envs/mkunet_mamba/bin/python scripts/run_pipeline.py \
  --config configs/reproduction.toml
```

## Notes

The ladder includes historical diagnostic scripts. Some stages are not meant to
be selected as the final model; they document failures and design decisions.
The public headline result remains `S27 X3 Final+B7`.
