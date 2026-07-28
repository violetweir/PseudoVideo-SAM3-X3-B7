# Reproduction Protocol

## Fixed Inputs

- Dataset: CVC-ClinicDB and Kvasir-SEG merged.
- Splits: `train=1290`, `validation=161`, `test=161`.
- Human supervision: exactly 16 train masks, listed in
  `protocols/reproduction_v1/support_ids.txt`.
- SAM3 is frozen for all route generation and B7 route candidates.
- Validation/test GT is evaluation-only.

## S27 X3 Mainline

1. Prepare path-local manifests from the fixed split protocol.
2. Run T21 `round0_train` with the 16 human anchors.
3. Select original 568 pseudo labels using `q_multi >= 0.90` and
   `q_return >= 0.95`.
4. Train/export committee auditors:
   `S2 val-best`, `S2 final`, and `S3 final`.
5. Audit the remaining 706 train images and freeze Tier A/B/C.
6. Build X3 from `568 original + 169 Tier A + 139 Tier B = 876` pseudo labels.
7. Train S27 X3 for 40000 steps.
8. Export X3 final predictions for train/validation/test.
9. Generate validation and test SAM3 route pools from the same 16 anchors.
10. Select routes with B7 and report final Test once.

## Quality Terms

- `q_return`: SAM3 return consistency. The predicted query mask is converted to
  a prompt and propagated back to the anchor. Dice with the original anchor mask
  measures whether target identity survived the pseudo-video path.
- `q_multi`: agreement between the three SAM3 route masks for the same query.
- `q_model`: Dice between a SAM3 route mask and the single-image X3 student mask.
- `B7`: geometric selector combining `q_return`, `q_multi`, and `q_model`.

## What Is Not Allowed

- Replacing the 16 anchors with validation/test images.
- Choosing support images, pseudo-label thresholds, route topology, or
  checkpoints using test masks.
- Feeding category text such as "polyp" into the route-generation protocol.
- Turning pseudo labels into new SAM3 propagation anchors in the S27 X3+B7
  mainline.
