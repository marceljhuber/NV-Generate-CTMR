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

Install dependencies first. For CUDA-enabled PyTorch, follow the PyTorch command matching the local CUDA driver, then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

Check OCT setup before training:

```bash
python -m scripts.check_oct_setup --check-wandb
```

If W&B is not needed for a smoke test, omit `--check-wandb`.

Run a minimal training smoke test before a full run:

```bash
python -m scripts.train_oct_vae \
  --max-train-batches 2 \
  --max-val-batches 1 \
  --num-recon-images 8
```

For W&B logging, first authenticate once:

```bash
wandb login
```

Train a 2D 128 px OCT VAE:

```bash
python -m scripts.train_oct_vae \
  --dataset-root /mnt/nas/media/ubuntu/data/ZhangLabData/CellData/OCT \
  --train-manifest ../../data_splits/kermanyv3_oct/train_manifest.csv \
  --val-manifest ../../data_splits/kermanyv3_oct/val_manifest.csv
```

For the first stable overnight run, prefer reconstruction-first full precision training:

```bash
python -m scripts.train_oct_vae \
  --train-config configs/config_maisi_vae_train_oct_128_stable.json \
  --model-dir models/oct_vae_128_stable \
  --output-dir outputs/oct_vae_128_stable \
  --num-recon-images 8 \
  --num-workers 4 \
  --no-amp \
  --wandb \
  --wandb-project oct-maisi
```

For a quick orientation-corrected checkpoint usable by the next pipeline stage, run the 5-epoch config:

```bash
python -m scripts.train_oct_vae \
  --train-config configs/config_maisi_vae_train_oct_128_5epochs.json \
  --model-dir models/oct_vae_128_5epochs \
  --output-dir outputs/oct_vae_128_5epochs \
  --num-recon-images 8 \
  --num-workers 4 \
  --no-amp \
  --wandb \
  --wandb-project oct-maisi
```

Create latents from a trained VAE checkpoint:

```bash
python -m scripts.create_oct_latents \
  --checkpoint models/oct_vae_128_5epochs/autoencoder_oct_128_best.pt \
  --output-dir outputs/oct_latents_128_5epochs \
  --batch-size 128 \
  --num-workers 4
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

The first W&B overnight run with AMP and adversarial loss became unstable and produced NaNs. Use the stable config above first: full precision, adversarial loss disabled, and NaN detection enabled.

## Current Readiness Checklist

- Dataset manifests exist and are patient-disjoint.
- OCT JPEG/PNG loading and transforms are implemented.
- OCT images are rotated 90 degrees clockwise during loading to match the expected visual orientation.
- W&B logging is wired through `--wandb`.
- Runtime verification still requires installing PyTorch and MONAI in the active environment.
- First training target is VAE reconstruction quality, not diffusion generation.
- The OCT VAE uses MONAI's generic 2D `AutoencoderKL`; NVIDIA's MAISI-specific autoencoder currently expects 5D volumetric tensors and is not suitable for direct 2D OCT training.

## VAE Tracking

The VAE trainer tracks:

- Train reconstruction loss, KL loss, SSIM loss, generator adversarial loss, and discriminator loss per step.
- Validation total loss, reconstruction loss, KL loss, SSIM loss, and SSIM score per validation epoch.
- Best checkpoint by validation total loss.
- Validation reconstruction grids under `outputs/oct_vae_128/reconstructions/`.

Each reconstruction grid has three rows:

```text
input OCT images
decoded reconstructions
absolute reconstruction error
```

By default, 8 validation images are saved per validation epoch. Increase `--num-recon-images` only if visual inspection needs more examples; 8 is enough to see whether the VAE preserves retinal layers without bloating logs.
