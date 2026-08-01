# Third-Party Dependencies

This repository contains integration and reproduction code plus the two
student-baseline codebases needed for the low-label comparisons.

It does not redistribute:

- SAM3 source code or checkpoints.
- CVC-ClinicDB images/masks.
- Kvasir-SEG images/masks.

Expected external paths are configured in `configs/reproduction.toml`.

## SAM3

SAM3 should be installed separately and run in a working SAM3 environment. The
original experiments used the conda environment named `sam3`.

## SC-SAM Student

SC-SAM is vendored at `third_party/SC-SAM` so T19 can be reproduced from this
repository. The historical checkout did not include a clear license file; keep
this in mind before public redistribution.

The student network imports:

- `Model.model.SamUnet`
- `dataloader.transforms`
- `dataloader.TwoStreamBatchSampler`
- `utils.losses`

Set `SC_SAM_ROOT` or `sc_sam_root` in the config if you want to override the
vendored checkout.

## SynFoC Student

The T20 experiment used a small metadata-backed copy of SynFoC adapted to the
merged CVC/Kvasir protocol. That exact runnable copy is included at
`third_party/SynFoC-T20` and is the default used by
`scripts/run_t20_synfoc_baseline.py`.

SynFoC includes an Apache-2.0 license.

## Datasets

Download CVC-ClinicDB and Kvasir-SEG from their official sources and follow
their licenses/citation requirements. This repository only stores path-free IDs
and split membership needed to reproduce the experiments.
