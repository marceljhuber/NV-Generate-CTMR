# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from monai.networks.schedulers import RFlowScheduler
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid, save_image

from .oct_data import OCT_ID_TO_LABEL
from .utils import define_instance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train 2D OCT latent diffusion with classifier-free guidance.")
    parser.add_argument("--latents-dir", type=Path, default=Path("outputs/oct_latents_128_5epochs"))
    parser.add_argument("--vae-checkpoint", type=Path, default=Path("models/oct_vae_128_5epochs/autoencoder_oct_128_best.pt"))
    parser.add_argument("--network-config", type=Path, default=Path("configs/config_network_oct_rflow.json"))
    parser.add_argument("--train-config", type=Path, default=Path("configs/config_maisi_diffusion_oct_128_50epochs.json"))
    parser.add_argument("--model-dir", type=Path, default=Path("models/oct_diffusion_128_50epochs"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/oct_diffusion_128_50epochs"))
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="oct-maisi")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def setup_wandb(enabled: bool, project: str, config: dict):
    if not enabled:
        return None
    import wandb

    return wandb.init(project=project, config=config)


def load_latent_split(latents_dir: Path, split: str) -> tuple[torch.Tensor, torch.Tensor]:
    manifest = load_json(latents_dir / f"{split}_shards.json")
    latents = []
    labels = []
    for shard in manifest["shards"]:
        data = torch.load(latents_dir / shard["path"], map_location="cpu", weights_only=False)
        shard_latents = data["latents"]
        if hasattr(shard_latents, "as_tensor"):
            shard_latents = shard_latents.as_tensor()
        latents.append(shard_latents.detach().float())
        shard_labels = data["class_labels"]
        if hasattr(shard_labels, "as_tensor"):
            shard_labels = shard_labels.as_tensor()
        labels.append(shard_labels.detach().long())
    return torch.cat(latents, dim=0).detach(), torch.cat(labels, dim=0).detach()


def make_loader(latents: torch.Tensor, labels: torch.Tensor, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(TensorDataset(latents, labels), batch_size=batch_size, shuffle=shuffle, num_workers=0, drop_last=shuffle)


def limited_batches(loader: DataLoader, max_batches: int | None):
    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        yield batch_idx, batch


def drop_labels(labels: torch.Tensor, dropout: float) -> torch.Tensor:
    if dropout <= 0:
        return labels
    keep = torch.rand(labels.shape, device=labels.device) >= dropout
    return labels * keep.long()


@torch.inference_mode()
def sample_images(
    unet: torch.nn.Module,
    autoencoder: torch.nn.Module,
    scheduler: RFlowScheduler,
    latent_shape: tuple[int, int, int],
    scale_factor: torch.Tensor,
    device: torch.device,
    samples_per_class: int,
    num_inference_steps: int,
    cfg_guidance_scale: float,
) -> torch.Tensor:
    unet.eval()
    autoencoder.eval()
    generated = []
    scheduler.set_timesteps(num_inference_steps=num_inference_steps, input_img_size_numel=torch.prod(torch.tensor(latent_shape[1:])))
    timesteps = scheduler.timesteps
    next_timesteps = torch.cat((timesteps[1:], torch.tensor([0], dtype=timesteps.dtype)))

    for class_id in [1, 2, 3, 4]:
        image = torch.randn((samples_per_class, *latent_shape), device=device)
        labels = torch.full((samples_per_class,), class_id, dtype=torch.long, device=device)
        for timestep, next_timestep in zip(timesteps, next_timesteps):
            t = torch.full((samples_per_class,), float(timestep), device=device)
            if cfg_guidance_scale > 0:
                model_input = torch.cat([image, image], dim=0)
                model_t = torch.cat([t, t], dim=0)
                model_labels = torch.cat([labels, torch.zeros_like(labels)], dim=0)
                pred_cond, pred_uncond = unet(model_input, timesteps=model_t, class_labels=model_labels).chunk(2)
                model_output = pred_uncond + cfg_guidance_scale * (pred_cond - pred_uncond)
            else:
                model_output = unet(image, timesteps=t, class_labels=labels)
            image, _ = scheduler.step(model_output, timestep, image, next_timestep)
        decoded = autoencoder.decode(image / scale_factor).detach().float().cpu().clamp(0, 1)
        generated.append(decoded)
    return torch.cat(generated, dim=0)


def save_generation_grid(images: torch.Tensor, output_path: Path, samples_per_class: int) -> torch.Tensor:
    grid = make_grid(images, nrow=samples_per_class, padding=2)
    save_image(grid, output_path)
    return grid


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for OCT diffusion training.")
    device = torch.device("cuda")
    args.model_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = args.output_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    network_config = load_json(args.network_config)
    train_config = load_json(args.train_config)["diffusion_train"]
    config_ns = argparse.Namespace(**network_config)
    unet = define_instance(config_ns, "diffusion_unet_def").to(device)
    autoencoder = define_instance(config_ns, "autoencoder_def").to(device)
    autoencoder.load_state_dict(torch.load(args.vae_checkpoint, map_location=device, weights_only=True))
    autoencoder.eval()
    scheduler = define_instance(config_ns, "noise_scheduler")

    train_latents, train_labels = load_latent_split(args.latents_dir, "train")
    val_latents, val_labels = load_latent_split(args.latents_dir, "val")
    scale_factor = (1.0 / train_latents.std()).to(device)
    latent_shape = tuple(train_latents.shape[1:])
    train_loader = make_loader(train_latents, train_labels, train_config["batch_size"], shuffle=True)
    val_loader = make_loader(val_latents, val_labels, train_config["batch_size"], shuffle=False)

    optimizer = torch.optim.Adam(unet.parameters(), lr=train_config["lr"])
    scaler = GradScaler("cuda", enabled=train_config["amp"])
    writer = SummaryWriter(log_dir=str(args.output_dir / "tfevents"))
    wandb_run = setup_wandb(args.wandb, args.wandb_project, {"network": network_config, "training": train_config})
    loss_fn = torch.nn.L1Loss()
    best_val = float("inf")
    global_step = 0

    for epoch in range(train_config["n_epochs"]):
        epoch_start = time.perf_counter()
        unet.train()
        train_loss = 0.0
        train_batches = 0
        batch_window_start = time.perf_counter()
        for batch_idx, (latents, labels) in limited_batches(train_loader, train_config.get("max_train_batches")):
            latents = latents.to(device) * scale_factor
            labels = drop_labels(labels.to(device), train_config["label_dropout"])
            optimizer.zero_grad(set_to_none=True)
            with autocast("cuda", enabled=train_config["amp"]):
                noise = torch.randn_like(latents)
                timesteps = scheduler.sample_timesteps(latents)
                noisy = scheduler.add_noise(original_samples=latents, noise=noise, timesteps=timesteps)
                pred = unet(noisy, timesteps=timesteps, class_labels=labels)
                target = latents - noise
                loss = loss_fn(pred.float(), target.float())
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite diffusion loss at epoch {epoch + 1}: {loss.item()}")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
            train_batches += 1
            global_step += 1
            writer.add_scalar("train/loss", loss.item(), global_step)
            if wandb_run:
                wandb_run.log({"train/loss": loss.item(), "global_step": global_step})
            if (batch_idx + 1) % 25 == 0:
                elapsed = time.perf_counter() - batch_window_start
                print(f"epoch {epoch + 1} train batch {batch_idx + 1}: loss={loss.item():.6f}, last_25_batches_sec={elapsed:.1f}", flush=True)
                batch_window_start = time.perf_counter()

        train_loss /= max(train_batches, 1)
        writer.add_scalar("epoch/train_loss", train_loss, epoch + 1)
        train_epoch_sec = time.perf_counter() - epoch_start

        if (epoch + 1) % train_config["val_interval"] == 0 or epoch + 1 == train_config["n_epochs"]:
            val_start = time.perf_counter()
            unet.eval()
            val_loss = 0.0
            with torch.no_grad():
                val_batches = 0
                for _, (latents, labels) in limited_batches(val_loader, train_config.get("max_val_batches")):
                    latents = latents.to(device) * scale_factor
                    labels = labels.to(device)
                    noise = torch.randn_like(latents)
                    timesteps = scheduler.sample_timesteps(latents)
                    noisy = scheduler.add_noise(original_samples=latents, noise=noise, timesteps=timesteps)
                    pred = unet(noisy, timesteps=timesteps, class_labels=labels)
                    val_loss += loss_fn(pred.float(), (latents - noise).float()).item()
                    val_batches += 1
            val_loss /= max(val_batches, 1)
            val_loss_sec = time.perf_counter() - val_start
            writer.add_scalar("val/loss", val_loss, epoch + 1)
            sample_start = time.perf_counter()
            sample_images_tensor = sample_images(
                unet,
                autoencoder,
                scheduler,
                latent_shape,
                scale_factor,
                device,
                train_config["samples_per_class"],
                train_config["num_inference_steps"],
                train_config["cfg_guidance_scale"],
            )
            sample_path = samples_dir / f"epoch_{epoch + 1:04d}_cfg{train_config['cfg_guidance_scale']}.png"
            grid = save_generation_grid(sample_images_tensor, sample_path, train_config["samples_per_class"])
            sample_sec = time.perf_counter() - sample_start
            writer.add_image("val/generated_by_class", grid, epoch + 1)
            metrics = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_epoch_sec": train_epoch_sec,
                "val_loss_sec": val_loss_sec,
                "sample_sec": sample_sec,
            }
            with (args.output_dir / "latest_metrics.json").open("w") as handle:
                json.dump(metrics, handle, indent=2)
                handle.write("\n")
            if val_loss < best_val:
                best_val = val_loss
                torch.save({"unet_state_dict": unet.state_dict(), "scale_factor": scale_factor.cpu(), "epoch": epoch + 1}, args.model_dir / "diffusion_oct_128_best.pt")
                with (args.output_dir / "best_metrics.json").open("w") as handle:
                    json.dump(metrics, handle, indent=2)
                    handle.write("\n")
            torch.save({"unet_state_dict": unet.state_dict(), "scale_factor": scale_factor.cpu(), "epoch": epoch + 1}, args.model_dir / "diffusion_oct_128_latest.pt")
            if wandb_run:
                import wandb

                wandb_run.log(
                    {
                        "epoch/train_loss": train_loss,
                        "val/loss": val_loss,
                        "timing/train_epoch_sec": train_epoch_sec,
                        "timing/val_loss_sec": val_loss_sec,
                        "timing/sample_sec": sample_sec,
                        "epoch": epoch + 1,
                        "val/generated_by_class": wandb.Image(str(sample_path), caption="rows: CNV, DME, DRUSEN, NORMAL"),
                    }
                )
            print(
                f"epoch {epoch + 1}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}, "
                f"train_epoch_sec={train_epoch_sec:.1f}, val_loss_sec={val_loss_sec:.1f}, sample_sec={sample_sec:.1f}",
                flush=True,
            )

    final_images = sample_images(
        unet,
        autoencoder,
        scheduler,
        latent_shape,
        scale_factor,
        device,
        train_config["final_samples_per_class"],
        train_config["num_inference_steps"],
        train_config["cfg_guidance_scale"],
    )
    final_path = samples_dir / f"final_25_per_class_cfg{train_config['cfg_guidance_scale']}.png"
    save_generation_grid(final_images, final_path, train_config["final_samples_per_class"])
    writer.close()
    if wandb_run:
        wandb_run.finish()


if __name__ == "__main__":
    main()
