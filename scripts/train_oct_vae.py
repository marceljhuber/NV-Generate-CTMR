# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch
from monai.data import CacheDataset, DataLoader
from monai.losses import SSIMLoss
from monai.networks.nets import PatchDiscriminator
from monai.transforms import Compose
from monai.utils import set_determinism
from torch.amp import GradScaler, autocast
from torch.nn import L1Loss, MSELoss
from torch.utils.tensorboard import SummaryWriter

from .oct_data import define_oct_image_transform, load_oct_manifest
from .utils import KL_loss, define_instance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a 2D OCT VAE for OCT-MAISI.")
    parser.add_argument("--dataset-root", type=Path, default=Path("/mnt/nas/media/ubuntu/data/ZhangLabData/CellData/OCT"))
    parser.add_argument("--train-manifest", type=Path, default=Path("../../data_splits/kermanyv3_oct/train_manifest.csv"))
    parser.add_argument("--val-manifest", type=Path, default=Path("../../data_splits/kermanyv3_oct/val_manifest.csv"))
    parser.add_argument("--network-config", type=Path, default=Path("configs/config_network_oct_rflow.json"))
    parser.add_argument("--train-config", type=Path, default=Path("configs/config_maisi_vae_train_oct_128.json"))
    parser.add_argument("--model-dir", type=Path, default=Path("models/oct_vae_128"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/oct_vae_128"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-train-batches", type=int, default=None, help="Optional smoke-test limit for train batches per epoch.")
    parser.add_argument("--max-val-batches", type=int, default=None, help="Optional smoke-test limit for validation batches.")
    parser.add_argument("--no-amp", dest="amp", action="store_false", help="Disable AMP for maximum numerical stability/debugging.")
    parser.add_argument("--wandb", action="store_true", help="Enable optional Weights & Biases logging.")
    parser.add_argument("--wandb-project", default="oct-maisi")
    parser.set_defaults(amp=True)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def setup_wandb(enabled: bool, project: str, config: dict):
    if not enabled:
        return None
    try:
        import wandb
    except ImportError as exc:
        raise ImportError("Install wandb or run without --wandb.") from exc
    return wandb.init(project=project, config=config)


def make_loader(records: list[dict], transform: Compose, cache_rate: float, batch_size: int, num_workers: int, shuffle: bool) -> DataLoader:
    dataset = CacheDataset(data=records, transform=transform, cache_rate=cache_rate, num_workers=num_workers)
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle, drop_last=shuffle)


def loss_weighted_sum(losses: dict[str, torch.Tensor], kl_weight: float, ssim_weight: float) -> torch.Tensor:
    return losses["recon_loss"] + kl_weight * losses["kl_loss"] + ssim_weight * losses["ssim_loss"]


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s][%(levelname)s] %(message)s")
    set_determinism(args.seed)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for OCT VAE training on this workstation.")
    device = torch.device("cuda")

    network_config = load_json(args.network_config)
    train_config = load_json(args.train_config)
    vae_train = train_config["autoencoder_train"]
    data_option = train_config["data_option"]

    config_ns = argparse.Namespace(**network_config)
    config_ns.model_dir = str(args.model_dir)
    args.model_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_records = load_oct_manifest(args.train_manifest, args.dataset_root)
    val_records = load_oct_manifest(args.val_manifest, args.dataset_root)
    logging.info("Loaded %d train and %d validation OCT records.", len(train_records), len(val_records))

    output_dtype = torch.float16 if args.amp else torch.float32
    train_transform = define_oct_image_transform(
        image_size=data_option["image_size"], is_train=True, output_dtype=output_dtype, random_aug=data_option["random_aug"]
    )
    val_transform = define_oct_image_transform(image_size=data_option["image_size"], is_train=False, output_dtype=output_dtype, random_aug=False)
    train_loader = make_loader(train_records, train_transform, vae_train["cache"], vae_train["batch_size"], args.num_workers, shuffle=True)
    val_loader = make_loader(val_records, val_transform, vae_train["cache"], vae_train["val_batch_size"], args.num_workers, shuffle=False)

    autoencoder = define_instance(config_ns, "autoencoder_def").to(device)
    discriminator = PatchDiscriminator(
        spatial_dims=network_config["spatial_dims"], num_layers_d=3, channels=32, in_channels=1, out_channels=1, norm="INSTANCE"
    ).to(device)

    recon_loss = MSELoss() if vae_train["recon_loss"] == "l2" else L1Loss(reduction="mean")
    ssim_loss = SSIMLoss(spatial_dims=2, data_range=1.0)
    optimizer_g = torch.optim.Adam(autoencoder.parameters(), lr=vae_train["lr"], eps=1e-6 if args.amp else 1e-8)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=vae_train["lr"], eps=1e-6 if args.amp else 1e-8)
    scaler_g = GradScaler("cuda", enabled=args.amp)
    scaler_d = GradScaler("cuda", enabled=args.amp)
    writer = SummaryWriter(log_dir=str(args.output_dir / "tfevents"))
    wandb_run = setup_wandb(
        args.wandb,
        args.wandb_project,
        {"network": network_config, "training": train_config, "seed": args.seed, "amp": args.amp},
    )

    best_val = float("inf")
    global_step = 0
    for epoch in range(vae_train["n_epochs"]):
        autoencoder.train()
        discriminator.train()
        epoch_losses = {"recon_loss": 0.0, "kl_loss": 0.0, "ssim_loss": 0.0}
        for batch_idx, batch in enumerate(train_loader):
            if args.max_train_batches is not None and batch_idx >= args.max_train_batches:
                break
            images = batch["image"].to(device).contiguous()
            optimizer_g.zero_grad(set_to_none=True)
            optimizer_d.zero_grad(set_to_none=True)

            with autocast("cuda", enabled=args.amp):
                reconstruction, z_mu, z_sigma = autoencoder(images)
                losses = {
                    "recon_loss": recon_loss(reconstruction, images),
                    "kl_loss": KL_loss(z_mu, z_sigma),
                    "ssim_loss": ssim_loss(reconstruction.float(), images.float()),
                }
                logits_fake = discriminator(reconstruction.contiguous().float())[-1]
                generator_adv = torch.mean((logits_fake - 1.0) ** 2)
                loss_g = loss_weighted_sum(losses, vae_train["kl_weight"], vae_train["ssim_weight"]) + vae_train["adv_weight"] * generator_adv

            scaler_g.scale(loss_g).backward()
            scaler_g.step(optimizer_g)
            scaler_g.update()

            with autocast("cuda", enabled=args.amp):
                logits_fake = discriminator(reconstruction.detach().contiguous().float())[-1]
                logits_real = discriminator(images.detach().contiguous().float())[-1]
                loss_d = 0.5 * (torch.mean(logits_fake**2) + torch.mean((logits_real - 1.0) ** 2))

            scaler_d.scale(loss_d).backward()
            scaler_d.step(optimizer_d)
            scaler_d.update()

            global_step += 1
            for key, value in losses.items():
                epoch_losses[key] += value.item()
                writer.add_scalar(f"train/{key}", value.item(), global_step)
            writer.add_scalar("train/generator_adv", generator_adv.item(), global_step)
            writer.add_scalar("train/discriminator_loss", loss_d.item(), global_step)
            if wandb_run:
                wandb_run.log({f"train/{key}": value.item() for key, value in losses.items()} | {"train/discriminator_loss": loss_d.item()})

        num_train_batches = max(min(len(train_loader), args.max_train_batches or len(train_loader)), 1)
        logging.info("Epoch %d train losses: %s", epoch + 1, {key: value / num_train_batches for key, value in epoch_losses.items()})

        if (epoch + 1) % vae_train["val_interval"] == 0:
            autoencoder.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_idx, batch in enumerate(val_loader):
                    if args.max_val_batches is not None and batch_idx >= args.max_val_batches:
                        break
                    images = batch["image"].to(device).contiguous()
                    with autocast("cuda", enabled=args.amp):
                        reconstruction, z_mu, z_sigma = autoencoder(images)
                        losses = {
                            "recon_loss": recon_loss(reconstruction, images),
                            "kl_loss": KL_loss(z_mu, z_sigma),
                            "ssim_loss": ssim_loss(reconstruction.float(), images.float()),
                        }
                        val_loss += loss_weighted_sum(losses, vae_train["kl_weight"], vae_train["ssim_weight"]).item()
            val_loss /= max(min(len(val_loader), args.max_val_batches or len(val_loader)), 1)
            writer.add_scalar("val/loss", val_loss, epoch + 1)
            if wandb_run:
                wandb_run.log({"val/loss": val_loss, "epoch": epoch + 1})
            logging.info("Epoch %d validation loss: %.6f", epoch + 1, val_loss)
            if val_loss < best_val:
                best_val = val_loss
                torch.save(autoencoder.state_dict(), args.model_dir / "autoencoder_oct_128_best.pt")

        torch.save(autoencoder.state_dict(), args.model_dir / "autoencoder_oct_128_latest.pt")
        torch.save(discriminator.state_dict(), args.model_dir / "discriminator_oct_128_latest.pt")

    writer.close()
    if wandb_run:
        wandb_run.finish()


if __name__ == "__main__":
    main()
