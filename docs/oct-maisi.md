# OCT-MAISI

This document tracks the OCT-specific adaptation path. The original CT/MR workflows remain unchanged.

## Dataset

The current dataset is KermanyV3 OCT, stored outside this repository:

```text
/mnt/nas/media/ubuntu/data/ZhangLabData/CellData/OCT/
```

Use fixed patient-level manifests from the thesis workspace:

```text
../../data_splits/kermanyv3_oct/train_manifest.csv
../../data_splits/kermanyv3_oct/val_manifest.csv
../../data_splits/kermanyv3_oct/test_manifest.csv
```

## Labels

OCT disease labels are used as class-conditioning labels:

| Label | ID |
|---|---:|
| `unknown` | 0 |
| `CNV` | 1 |
| `DME` | 2 |
| `DRUSEN` | 3 |
| `NORMAL` | 4 |

## First Training Step: VAE

Train a 2D 128 px OCT VAE:

```bash
python -m scripts.train_oct_vae \
  --dataset-root /mnt/nas/media/ubuntu/data/ZhangLabData/CellData/OCT \
  --train-manifest ../../data_splits/kermanyv3_oct/train_manifest.csv \
  --val-manifest ../../data_splits/kermanyv3_oct/val_manifest.csv
```

Optional W&B logging:

```bash
python -m scripts.train_oct_vae --wandb --wandb-project oct-maisi
```

Disable AMP for maximum numerical stability/debugging:

```bash
python -m scripts.train_oct_vae --no-amp
```

## Precision Policy

AMP is enabled by default for speed on the RTX 4070 Ti SUPER. For final quality-sensitive runs, compare at least one short AMP run against `--no-amp`. Keep the best validation/reconstruction behavior, not the fastest setting.
