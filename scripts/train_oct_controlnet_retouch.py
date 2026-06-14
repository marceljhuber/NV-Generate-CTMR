# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from monai.networks.nets import ControlNet
from monai.networks.utils import copy_model_state
from PIL import Image
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid, save_image

from .utils import define_instance


FLUID_LABELS = {1: "IRF", 2: "SRF", 3: "PED"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train 2D OCT ControlNet on RETOUCH fluid masks.")
    parser.add_argument("--manifest", type=Path, default=Path("/mnt/nas/media/ubuntu/Thesis/maisi-v2/data/retouch/png_fullres/slice_manifest.csv"))
    parser.add_argument("--network-config", type=Path, default=Path("configs/config_network_oct_rflow.json"))
    parser.add_argument("--train-config", type=Path, default=Path("configs/config_oct_controlnet_retouch_256_smoke.json"))
    parser.add_argument("--vae-checkpoint", type=Path, default=Path("models/oct_vae_256_adv_overnight/autoencoder_oct_256_best.pt"))
    parser.add_argument("--diffusion-checkpoint", type=Path, default=Path("models/oct_diffusion_256_adv_overnight/diffusion_oct_256_best.pt"))
    parser.add_argument("--model-dir", type=Path, default=Path("models/oct_controlnet_retouch_256_smoke"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/oct_controlnet_retouch_256_smoke"))
    parser.add_argument("--splits", nargs="+", default=["train", "test_muw"], help="RETOUCH splits to train on. Default follows the thesis decision: train + MUW test labels.")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="oct-maisi")
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def setup_wandb(enabled: bool, project: str, name: str | None, config: dict):
    if not enabled:
        return None
    import wandb

    run = wandb.init(project=project, name=name, config=config)
    run.define_metric("epoch")
    run.define_metric("epoch/*", step_metric="epoch")
    run.define_metric("timing/*", step_metric="epoch")
    return run


def load_retouch_records(manifest: Path, splits: set[str]) -> list[dict]:
    records = []
    with manifest.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row["split"] not in splits:
                continue
            if "has_fluid" in row and row["has_fluid"] != "":
                has_fluid = row["has_fluid"].lower() in {"1", "true", "yes"}
                label_values = row.get("label_values", "")
            else:
                # Avoid a slow full-NAS mask scan at startup. If users want weighted
                # foreground sampling, pass a manifest with precomputed has_fluid.
                has_fluid = True
                label_values = ""
            records.append(row | {"has_fluid": has_fluid, "label_values": label_values})
    if not records:
        raise ValueError(f"No RETOUCH records found in {manifest} for splits={sorted(splits)}")
    return records


def mask_unique_values(path: Path) -> set[int]:
    with Image.open(path) as image:
        mask = np.asarray(image)
    return set(int(value) for value in np.unique(mask))


def resize_image_array(array: np.ndarray, image_size: int, is_mask: bool) -> np.ndarray:
    mode = Image.Resampling.NEAREST if is_mask else Image.Resampling.BILINEAR
    return np.asarray(Image.fromarray(array).resize((image_size, image_size), mode))


class RetouchControlNetDataset(Dataset):
    def __init__(self, records: list[dict], image_size: int, augment: bool = True) -> None:
        self.records = records
        self.image_size = image_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str | bool]:
        record = self.records[index]
        with Image.open(record["image_path"]) as image_file:
            image = np.asarray(image_file.convert("L"))
        with Image.open(record["label_mask_path"]) as mask_file:
            label_mask = np.asarray(mask_file.convert("L"))

        # Match the OCT orientation used by the Kermany-trained 256 VAE/diffusion.
        image = np.rot90(image, k=-1).copy()
        label_mask = np.rot90(label_mask, k=-1).copy()
        # RETOUCH Spectralis volumes use a white-background convention, while
        # Cirrus/Topcon are already close to the Kermany dark-background OCT convention.
        if record["scanner"] == "Spectralis":
            image = 255 - image

        image = resize_image_array(image, self.image_size, is_mask=False).astype(np.float32) / 255.0
        label_mask = resize_image_array(label_mask, self.image_size, is_mask=True).astype(np.int64)

        if self.augment and random.random() < 0.5:
            image = np.flip(image, axis=1).copy()
            label_mask = np.flip(label_mask, axis=1).copy()

        cond = np.stack([(label_mask == label_value).astype(np.float32) for label_value in FLUID_LABELS], axis=0)
        return {
            "image": torch.from_numpy(image).unsqueeze(0),
            "mask": torch.from_numpy(cond),
            "class_label": torch.tensor(0, dtype=torch.long),
            "has_fluid": bool(record["has_fluid"]),
            "image_path": record["image_path"],
            "label_mask_path": record["label_mask_path"],
        }


def make_loader(records: list[dict], train_config: dict) -> DataLoader:
    dataset = RetouchControlNetDataset(records, image_size=train_config["image_size"], augment=True)
    weights = [train_config["foreground_sample_weight"] if record["has_fluid"] else train_config["empty_sample_weight"] for record in records]
    sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
    return DataLoader(
        dataset,
        batch_size=train_config["batch_size"],
        sampler=sampler,
        num_workers=train_config["num_workers"],
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        persistent_workers=train_config["num_workers"] > 0,
    )


def build_controlnet(network_config: dict) -> ControlNet:
    unet_def = network_config["diffusion_unet_def"]
    return ControlNet(
        spatial_dims=network_config["spatial_dims"],
        in_channels=network_config["latent_channels"],
        channels=unet_def["channels"],
        attention_levels=unet_def["attention_levels"],
        num_head_channels=unet_def["num_head_channels"],
        num_res_blocks=unet_def["num_res_blocks"],
        resblock_updown=unet_def["resblock_updown"],
        num_class_embeds=unet_def["num_class_embeds"],
        conditioning_embedding_in_channels=len(FLUID_LABELS),
        conditioning_embedding_num_channels=(16, 32, 64),
        include_fc=unet_def["include_fc"],
        use_flash_attention=unet_def.get("use_flash_attention", False),
    )


def maybe_drop_masks(masks: torch.Tensor, dropout: float) -> torch.Tensor:
    if dropout <= 0:
        return masks
    keep = (torch.rand((masks.shape[0], 1, 1, 1), device=masks.device) >= dropout).to(masks.dtype)
    return masks * keep


@torch.inference_mode()
def sample_controlnet_images(
    autoencoder: torch.nn.Module,
    unet: torch.nn.Module,
    controlnet: ControlNet,
    scheduler,
    masks: torch.Tensor,
    scale_factor: torch.Tensor,
    train_config: dict,
    device: torch.device,
) -> torch.Tensor:
    autoencoder.eval()
    unet.eval()
    controlnet.eval()
    masks = masks.to(device)
    batch_size = masks.shape[0]
    latent_shape = (batch_size, 4, train_config["image_size"] // 4, train_config["image_size"] // 4)
    image = torch.randn(latent_shape, device=device)
    labels = torch.zeros((batch_size,), dtype=torch.long, device=device)
    scheduler.set_timesteps(num_inference_steps=train_config["num_inference_steps"], input_img_size_numel=(train_config["image_size"] // 4) ** 2)
    timesteps = scheduler.timesteps
    next_timesteps = torch.cat((timesteps[1:], torch.tensor([0], dtype=timesteps.dtype)))
    for timestep, next_timestep in zip(timesteps, next_timesteps):
        t = torch.full((batch_size,), float(timestep), device=device)
        down, mid = controlnet(image, timesteps=t, controlnet_cond=masks, class_labels=labels, conditioning_scale=train_config["conditioning_scale"])
        model_output = unet(image, timesteps=t, class_labels=labels, down_block_additional_residuals=down, mid_block_additional_residual=mid)
        image, _ = scheduler.step(model_output, timestep, image, next_timestep)
    return autoencoder.decode(image / scale_factor).detach().float().cpu().clamp(0, 1)


def make_mask_grid(masks: torch.Tensor) -> torch.Tensor:
    colors = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.25, 1.0]], dtype=torch.float32)
    rgb = torch.einsum("bchw,cr->brhw", masks.float().cpu(), colors).clamp(0, 1)
    return rgb


def write_metrics(path: Path, metrics: dict) -> None:
    with path.open("w") as handle:
        json.dump(metrics, handle, indent=2)
        handle.write("\n")


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for OCT ControlNet training.")
    device = torch.device("cuda")
    args.model_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = args.output_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    network_config = load_json(args.network_config)
    train_config = load_json(args.train_config)["controlnet_train"]
    config_ns = argparse.Namespace(**network_config)
    autoencoder = define_instance(config_ns, "autoencoder_def").to(device)
    unet = define_instance(config_ns, "diffusion_unet_def").to(device)
    scheduler = define_instance(config_ns, "noise_scheduler")
    autoencoder.load_state_dict(torch.load(args.vae_checkpoint, map_location=device, weights_only=True))
    diffusion_ckpt = torch.load(args.diffusion_checkpoint, map_location=device, weights_only=False)
    unet.load_state_dict(diffusion_ckpt["unet_state_dict"])
    scale_factor = diffusion_ckpt["scale_factor"].to(device)
    controlnet = build_controlnet(network_config).to(device)
    copy_model_state(controlnet, unet.state_dict())
    for module in (autoencoder, unet):
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad = False

    records = load_retouch_records(args.manifest, set(args.splits))
    fluid_slices = sum(1 for record in records if record["has_fluid"])
    print(f"Loaded {len(records)} RETOUCH B-scans from splits={args.splits}; fluid_slices={fluid_slices}, empty_slices={len(records) - fluid_slices}")
    loader = make_loader(records, train_config)
    sample_dataset = RetouchControlNetDataset([record for record in records if record["has_fluid"]][: train_config["num_sample_masks"]], train_config["image_size"], augment=False)
    sample_masks = torch.stack([sample_dataset[index]["mask"] for index in range(len(sample_dataset))])

    optimizer = torch.optim.AdamW(controlnet.parameters(), lr=train_config["lr"])
    scaler = GradScaler("cuda", enabled=train_config["amp"])
    writer = SummaryWriter(log_dir=str(args.output_dir / "tfevents"))
    wandb_run = setup_wandb(
        args.wandb,
        args.wandb_project,
        args.wandb_name,
        {"network": network_config, "training": train_config, "splits": args.splits, "records": len(records), "fluid_slices": fluid_slices},
    )
    best_loss = float("inf")
    epochs_without_improvement = 0
    early_stop_patience = train_config.get("early_stop_patience")
    early_stop_min_delta = train_config.get("early_stop_min_delta", 0.0)
    global_step = 0
    checkpoint_prefix = train_config["checkpoint_prefix"]
    for epoch in range(train_config["n_epochs"]):
        epoch_start = time.perf_counter()
        controlnet.train()
        epoch_loss = 0.0
        num_batches = 0
        for batch_idx, batch in enumerate(loader):
            if train_config["max_train_batches"] is not None and batch_idx >= train_config["max_train_batches"]:
                break
            images = batch["image"].to(device)
            masks = maybe_drop_masks(batch["mask"].to(device), train_config["mask_dropout"])
            class_labels = batch["class_label"].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad(), autocast("cuda", enabled=train_config["amp"]):
                z_mu, _ = autoencoder.encode(images)
                latents = z_mu * scale_factor
            with autocast("cuda", enabled=train_config["amp"]):
                noise = torch.randn_like(latents)
                timesteps = scheduler.sample_timesteps(latents)
                noisy = scheduler.add_noise(original_samples=latents, noise=noise, timesteps=timesteps)
                down, mid = controlnet(noisy, timesteps=timesteps, controlnet_cond=masks, class_labels=class_labels, conditioning_scale=train_config["conditioning_scale"])
                pred = unet(noisy, timesteps=timesteps, class_labels=class_labels, down_block_additional_residuals=down, mid_block_additional_residual=mid)
                loss = F.l1_loss(pred.float(), (latents - noise).float())
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite ControlNet loss at epoch {epoch + 1}, batch {batch_idx + 1}: {loss.item()}")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()
            num_batches += 1
            global_step += 1
            writer.add_scalar("train/loss", loss.item(), global_step)
            if wandb_run:
                wandb_run.log({"train/loss": loss.item(), "global_step": global_step})

        epoch_loss /= max(num_batches, 1)
        epoch_sec = time.perf_counter() - epoch_start
        writer.add_scalar("epoch/train_loss", epoch_loss, epoch + 1)
        metrics = {"epoch": epoch + 1, "train_loss": epoch_loss, "epoch_sec": epoch_sec, "global_step": global_step}
        write_metrics(args.output_dir / "latest_metrics.json", metrics)
        controlnet_state = controlnet.state_dict()
        torch.save({"epoch": epoch + 1, "loss": epoch_loss, "controlnet_state_dict": controlnet_state, "scale_factor": scale_factor.detach().cpu(), "train_config": train_config}, args.model_dir / f"{checkpoint_prefix}_latest.pt")
        if epoch_loss < best_loss - early_stop_min_delta:
            best_loss = epoch_loss
            epochs_without_improvement = 0
            torch.save({"epoch": epoch + 1, "loss": best_loss, "controlnet_state_dict": controlnet_state, "scale_factor": scale_factor.detach().cpu(), "train_config": train_config}, args.model_dir / f"{checkpoint_prefix}_best.pt")
            write_metrics(args.output_dir / "best_metrics.json", metrics)
        else:
            epochs_without_improvement += 1

        sample_path = None
        mask_path = None
        if (epoch + 1) % train_config["sample_interval"] == 0 and len(sample_masks) > 0:
            generated = sample_controlnet_images(autoencoder, unet, controlnet, scheduler, sample_masks, scale_factor, train_config, device)
            sample_path = samples_dir / f"epoch_{epoch + 1:04d}_generated.png"
            mask_path = samples_dir / f"epoch_{epoch + 1:04d}_masks.png"
            save_image(make_grid(generated, nrow=len(generated), padding=2), sample_path)
            save_image(make_grid(make_mask_grid(sample_masks), nrow=len(sample_masks), padding=2), mask_path)

        if wandb_run:
            log_data = {"epoch": epoch + 1, "epoch/train_loss": epoch_loss, "timing/epoch_sec": epoch_sec}
            if sample_path is not None:
                import wandb

                log_data["samples/generated"] = wandb.Image(str(sample_path))
                log_data["samples/masks"] = wandb.Image(str(mask_path))
            wandb_run.log(log_data)
        print(f"epoch {epoch + 1}: train_loss={epoch_loss:.6f}, epoch_sec={epoch_sec:.1f}", flush=True)
        if early_stop_patience is not None and epochs_without_improvement >= early_stop_patience:
            print(
                f"early stopping after {epochs_without_improvement} epochs without improvement; best_loss={best_loss:.6f}",
                flush=True,
            )
            break

    writer.close()
    if wandb_run:
        wandb_run.finish()


if __name__ == "__main__":
    main()
