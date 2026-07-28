# Third-Party Dependencies

This repository contains integration and reproduction code only.

It does not redistribute:

- SAM3 source code or checkpoints.
- SC-SAM source code or checkpoints.
- CVC-ClinicDB images/masks.
- Kvasir-SEG images/masks.

Expected external paths are configured in `configs/reproduction.toml`.

## SAM3

SAM3 should be installed separately and run in a working SAM3 environment. The
original experiments used the conda environment named `sam3`.

## SC-SAM Student

The student network imports:

- `Model.model.SamUnet`
- `dataloader.transforms`
- `dataloader.TwoStreamBatchSampler`
- `utils.losses`

Set `SC_SAM_ROOT` or `sc_sam_root` in the config to a compatible checkout. The
historical local checkout did not include a clear redistributable license, so it
is not vendored here.

## Datasets

Download CVC-ClinicDB and Kvasir-SEG from their official sources and follow
their licenses/citation requirements. This repository only stores path-free IDs
and split membership needed to reproduce the experiments.
