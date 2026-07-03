# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image

from .train_oct_controlnet_retouch import RetouchControlNetDataset, make_mask_grid, tensor_stats
from .utils import define_instance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dif-fuse-style OCT counterfactual removal with rectified-flow latent inversion.")
    parser.add_argument("--retouch-manifest", type=Path, default=Path("/mnt/nas/media/ubuntu/Thesis/maisi-v2/data/retouch/analysis/retouch_controlnet_spectralis_train_muw_manifest.csv"))
    parser.add_argument("--network-config", type=Path, default=Path("configs/config_network_oct_rflow.json"))
    parser.add_argument("--vae-checkpoint", type=Path, default=Path("models/oct_vae_256_adv_overnight/autoencoder_oct_256_best.pt"))
    parser.add_argument("--diffusion-checkpoint", type=Path, default=Path("models/oct_diffusion_256_adv_overnight/diffusion_oct_256_best.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/oct_diffuse_counterfactuals_256_smoke"))
    parser.add_argument("--num-images", type=int, default=12)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--num-inference-steps", type=int, default=75)
    parser.add_argument("--inversion-fractions", type=float, nargs="+", default=[0.35, 0.5, 0.65])
    parser.add_argument("--mask-dilate", type=int, default=5, help="Odd-pixel dilation kernel applied to the union fluid mask before latent downsampling.")
    parser.add_argument("--edit-label", type=int, default=4, help="Class label used as healthy prior. Kermany NORMAL is 4.")
    parser.add_argument("--reconstruct-label", type=int, default=4, help="Class label used for reconstruction/inversion. Default NORMAL for pseudo-healthy prior.")
    parser.add_argument("--masked-randomize", action="store_true", help="Replace masked latent area with random noise at the selected inversion depth before denoising.")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--grid-chunk-size", type=int, default=6)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def load_retouch_records(manifest: Path) -> list[dict]:
    records = []
    with manifest.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            has_fluid = row.get("has_fluid", "0").lower() in {"1", "true", "yes"}
            if row.get("scanner") == "Spectralis" and row.get("split") in {"train", "test_muw"} and has_fluid:
                records.append(row | {"has_fluid": has_fluid})
    if not records:
        raise ValueError(f"No fluid-positive Spectralis RETOUCH records found in {manifest}")
    return records


def write_json(path: Path, data: dict) -> None:
    with path.open("w") as handle:
        json.dump(data, handle, indent=2)
        handle.write("\n")


def load_models(args: argparse.Namespace, device: torch.device):
    network_config = load_json(args.network_config)
    config_ns = argparse.Namespace(**network_config)
    autoencoder = define_instance(config_ns, "autoencoder_def").to(device)
    unet = define_instance(config_ns, "diffusion_unet_def").to(device)
    scheduler = define_instance(config_ns, "noise_scheduler")

    autoencoder.load_state_dict(torch.load(args.vae_checkpoint, map_location=device, weights_only=True))
    diffusion_ckpt = torch.load(args.diffusion_checkpoint, map_location=device, weights_only=False)
    unet.load_state_dict(diffusion_ckpt["unet_state_dict"])
    scale_factor = diffusion_ckpt["scale_factor"].to(device)
    autoencoder.eval()
    unet.eval()
    return autoencoder, unet, scheduler, scale_factor


def fluid_union_mask(masks: torch.Tensor, dilate: int) -> torch.Tensor:
    union = (masks.sum(dim=1, keepdim=True) > 0).float()
    if dilate > 1:
        if dilate % 2 == 0:
            raise ValueError("--mask-dilate must be odd so padding keeps image size unchanged.")
        union = F.max_pool2d(union, kernel_size=dilate, stride=1, padding=dilate // 2)
    return union


def latent_mask_from_image_mask(mask: torch.Tensor, latent_hw: tuple[int, int]) -> torch.Tensor:
    return F.interpolate(mask, size=latent_hw, mode="nearest")


def model_velocity(unet: torch.nn.Module, latents: torch.Tensor, timestep: torch.Tensor | float, label: int) -> torch.Tensor:
    labels = torch.full((latents.shape[0],), label, dtype=torch.long, device=latents.device)
    if not torch.is_tensor(timestep):
        timestep = torch.full((latents.shape[0],), float(timestep), device=latents.device)
    return unet(latents, timesteps=timestep, class_labels=labels)


@torch.inference_mode()
def invert_rflow(unet: torch.nn.Module, clean_latents: torch.Tensor, timesteps: torch.Tensor, reconstruct_label: int) -> torch.Tensor:
    """Approximate rectified-flow inversion by integrating from clean latents toward higher noise timesteps."""
    latents = clean_latents.clone()
    forward_timesteps = torch.flip(timesteps, dims=[0])
    for current_t, next_t in zip(forward_timesteps[:-1], forward_timesteps[1:]):
        t = torch.full((latents.shape[0],), float(current_t), device=latents.device)
        velocity = model_velocity(unet, latents, t, reconstruct_label)
        dt = float(next_t - current_t) / 1000.0
        latents = latents - velocity * dt
    return latents


@torch.inference_mode()
def denoise_rflow(unet: torch.nn.Module, scheduler, start_latents: torch.Tensor, timesteps: torch.Tensor, label: int) -> torch.Tensor:
    latents = start_latents.clone()
    next_timesteps = torch.cat((timesteps[1:], torch.tensor([0], dtype=timesteps.dtype, device=timesteps.device)))
    for timestep, next_timestep in zip(timesteps, next_timesteps):
        t = torch.full((latents.shape[0],), float(timestep), device=latents.device)
        velocity = model_velocity(unet, latents, t, label)
        latents, _ = scheduler.step(velocity, timestep, latents, next_timestep)
    return latents


@torch.inference_mode()
def diffuse_masked_edit(
    unet: torch.nn.Module,
    scheduler,
    clean_latents: torch.Tensor,
    latent_mask: torch.Tensor,
    timesteps: torch.Tensor,
    edit_label: int,
    reconstruct_label: int,
    masked_randomize: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    inverted = invert_rflow(unet, clean_latents, timesteps, reconstruct_label)
    current = inverted.clone()
    if masked_randomize:
        current = torch.randn_like(current) * latent_mask + current * (1.0 - latent_mask)

    next_timesteps = torch.cat((timesteps[1:], torch.tensor([0], dtype=timesteps.dtype, device=timesteps.device)))
    for timestep, next_timestep in zip(timesteps, next_timesteps):
        t = torch.full((current.shape[0],), float(timestep), device=current.device)

        recon_velocity = model_velocity(unet, current, t, reconstruct_label)
        edit_velocity = model_velocity(unet, current, t, edit_label)
        recon_step, _ = scheduler.step(recon_velocity, timestep, current, next_timestep)
        edit_step, _ = scheduler.step(edit_velocity, timestep, current, next_timestep)
        current = edit_step * latent_mask + recon_step * (1.0 - latent_mask)
    return current, inverted


@torch.inference_mode()
def decode(autoencoder: torch.nn.Module, latents: torch.Tensor, scale_factor: torch.Tensor) -> torch.Tensor:
    return autoencoder.decode(latents / scale_factor).detach().float().cpu().clamp(0, 1)


def save_review_grids(args: argparse.Namespace, images: torch.Tensor, masks: torch.Tensor, reconstructions: dict[str, torch.Tensor], edits: dict[str, torch.Tensor]) -> None:
    def rgb(tensor: torch.Tensor) -> torch.Tensor:
        tensor = tensor.cpu()
        return tensor.repeat(1, 3, 1, 1) if tensor.shape[1] == 1 else tensor

    mask_rgb = make_mask_grid(masks.cpu())
    for name, edited in edits.items():
        diff = (images.cpu() - edited.cpu()).abs()
        rows = [rgb(images), mask_rgb, rgb(reconstructions[name]), rgb(edited), rgb(diff)]
        row_names = ["original", "fluid_mask", "reconstruction_control", "masked_counterfactual", "absolute_difference"]
        save_image(make_grid(torch.cat(rows, dim=0), nrow=args.num_images, padding=2), args.output_dir / f"review_{name}_all.png")
        for start in range(0, args.num_images, args.grid_chunk_size):
            end = min(start + args.grid_chunk_size, args.num_images)
            chunk = torch.cat([row[start:end] for row in rows], dim=0)
            save_image(make_grid(chunk, nrow=end - start, padding=2), args.output_dir / f"review_{name}_{start:03d}_{end - 1:03d}.png")
        (args.output_dir / f"review_{name}_row_order.txt").write_text("\n".join(row_names) + "\n")


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for OCT Dif-fuse counterfactual generation.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda")

    records = load_retouch_records(args.retouch_manifest)
    chosen = random.sample(records, min(args.num_images, len(records)))
    dataset = RetouchControlNetDataset(chosen, image_size=args.image_size, augment=False)
    images = torch.stack([dataset[index]["image"] for index in range(len(chosen))])
    masks = torch.stack([dataset[index]["mask"] for index in range(len(chosen))])

    autoencoder, unet, scheduler, scale_factor = load_models(args, device)
    images_device = images.to(device)
    z_mu, _ = autoencoder.encode(images_device)
    clean_latents = z_mu * scale_factor
    image_mask = fluid_union_mask(masks.to(device), args.mask_dilate)
    latent_mask = latent_mask_from_image_mask(image_mask, clean_latents.shape[-2:])

    scheduler.set_timesteps(num_inference_steps=args.num_inference_steps, device=device, input_img_size_numel=clean_latents.shape[-1] * clean_latents.shape[-2])
    full_timesteps = scheduler.timesteps

    reconstructions: dict[str, torch.Tensor] = {}
    edits: dict[str, torch.Tensor] = {}
    metrics: dict[str, object] = {
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "records": chosen,
        "inputs": tensor_stats(images),
        "masks": tensor_stats(masks),
        "union_mask": tensor_stats(image_mask.cpu()),
        "latent_mask": tensor_stats(latent_mask.cpu()),
        "runs": {},
    }

    save_image(make_grid(images.cpu(), nrow=args.grid_chunk_size, padding=2), args.output_dir / "inputs_retouch.png")
    save_image(make_grid(make_mask_grid(masks.cpu()), nrow=args.grid_chunk_size, padding=2), args.output_dir / "retouch_fluid_masks.png")
    save_image(make_grid(image_mask.cpu(), nrow=args.grid_chunk_size, padding=2), args.output_dir / "union_binary_masks.png")

    for fraction in args.inversion_fractions:
        start_count = max(2, min(len(full_timesteps), int(round(len(full_timesteps) * fraction))))
        timesteps = full_timesteps[-start_count:]
        name = f"frac{fraction:g}_steps{start_count}"
        inverted = invert_rflow(unet, clean_latents, timesteps, args.reconstruct_label)
        recon_latents = denoise_rflow(unet, scheduler, inverted, timesteps, args.reconstruct_label)
        edit_latents, _ = diffuse_masked_edit(unet, scheduler, clean_latents, latent_mask, timesteps, args.edit_label, args.reconstruct_label, args.masked_randomize)

        recon = decode(autoencoder, recon_latents, scale_factor)
        edited = decode(autoencoder, edit_latents, scale_factor)
        reconstructions[name] = recon
        edits[name] = edited
        save_image(make_grid(recon, nrow=args.grid_chunk_size, padding=2), args.output_dir / f"reconstruction_control_{name}.png")
        save_image(make_grid(edited, nrow=args.grid_chunk_size, padding=2), args.output_dir / f"masked_counterfactual_{name}.png")
        save_image(make_grid((images.cpu() - edited).abs(), nrow=args.grid_chunk_size, padding=2), args.output_dir / f"absolute_difference_{name}.png")

        mask_cpu = image_mask.cpu()
        outside = 1.0 - mask_cpu
        eps = 1e-6
        outside_l1 = float(((images.cpu() - edited).abs() * outside).sum() / (outside.sum() + eps))
        inside_l1 = float(((images.cpu() - edited).abs() * mask_cpu).sum() / (mask_cpu.sum() + eps))
        recon_l1 = float((images.cpu() - recon).abs().mean())
        metrics["runs"][name] = {
            "inversion_fraction": fraction,
            "num_reverse_steps": start_count,
            "start_timestep": float(timesteps[0]),
            "reconstruction_l1": recon_l1,
            "counterfactual_inside_mask_l1": inside_l1,
            "counterfactual_outside_mask_l1": outside_l1,
            "reconstruction": tensor_stats(recon),
            "counterfactual": tensor_stats(edited),
            "absolute_difference": tensor_stats((images.cpu() - edited).abs()),
        }

    save_review_grids(args, images, masks, reconstructions, edits)
    write_json(args.output_dir / "metrics.json", metrics)
    print(json.dumps({"output_dir": str(args.output_dir), "runs": metrics["runs"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
