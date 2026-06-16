# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import torch
from torchvision.utils import make_grid, save_image

from .oct_data import load_oct_manifest, define_oct_image_transform
from .train_oct_controlnet_retouch import RetouchControlNetDataset, build_controlnet, make_mask_grid, tensor_stats, write_metrics
from .utils import define_instance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate OCT ControlNet mask-conditioned samples and image-to-image counterfactuals.")
    parser.add_argument("--retouch-manifest", type=Path, default=Path("/mnt/nas/media/ubuntu/Thesis/maisi-v2/data/retouch/analysis/retouch_controlnet_spectralis_train_muw_manifest.csv"))
    parser.add_argument("--kermany-manifest", type=Path, default=Path("/mnt/nas/media/ubuntu/Thesis/maisi-v2/data_splits/kermanyv3_oct/train_manifest.csv"))
    parser.add_argument("--kermany-root", type=Path, default=Path("/home/user/oct-maisi-cache/OCT"))
    parser.add_argument("--network-config", type=Path, default=Path("configs/config_network_oct_rflow.json"))
    parser.add_argument("--vae-checkpoint", type=Path, default=Path("models/oct_vae_256_adv_overnight/autoencoder_oct_256_best.pt"))
    parser.add_argument("--diffusion-checkpoint", type=Path, default=Path("models/oct_diffusion_256_adv_overnight/diffusion_oct_256_best.pt"))
    parser.add_argument("--controlnet-checkpoint", type=Path, default=Path("models/oct_controlnet_retouch_256_corrected_png_long_geom_aug_bs32/controlnet_oct_retouch_256_long_geom_aug_best.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/oct_controlnet_counterfactuals_geom_best"))
    parser.add_argument("--num-images", type=int, default=50)
    parser.add_argument("--num-inference-steps", type=int, default=75)
    parser.add_argument("--strengths", type=float, nargs="+", default=[0.25, 0.5, 0.75])
    parser.add_argument("--conditioning-scale", type=float, default=1.0)
    parser.add_argument("--class-label", type=int, default=0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--grid-chunk-size", type=int, default=10)
    parser.add_argument("--input-source", choices=["kermany-normal", "retouch-empty-neighbor"], default="kermany-normal")
    parser.add_argument("--max-neighbor-distance", type=int, default=8)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def load_retouch_records(manifest: Path) -> list[dict]:
    records = []
    with manifest.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            has_fluid = row.get("has_fluid", "1").lower() in {"1", "true", "yes"}
            records.append(row | {"has_fluid": has_fluid})
    return records


def select_kermany_inputs(args: argparse.Namespace) -> tuple[torch.Tensor, list[dict]]:
    records = load_oct_manifest(args.kermany_manifest, args.kermany_root)
    normal_records = [record for record in records if record["label"] == "NORMAL"]
    if len(normal_records) < args.num_images:
        raise ValueError(f"Requested {args.num_images} NORMAL Kermany images, found {len(normal_records)}")
    chosen = random.sample(normal_records, args.num_images)
    transform = define_oct_image_transform(256, is_train=False, random_aug=False)
    images = torch.stack([transform(record)["image"] for record in chosen])
    return images, chosen


def select_retouch_empty_neighbor_pairs(args: argparse.Namespace) -> tuple[torch.Tensor, torch.Tensor, list[dict], list[dict]]:
    retouch_records = load_retouch_records(args.retouch_manifest)
    by_case: dict[str, list[dict]] = {}
    for record in retouch_records:
        if record.get("scanner") != "Spectralis" or record.get("split") not in {"train", "test_muw"}:
            continue
        by_case.setdefault(record["case_id"], []).append(record)

    pairs: list[tuple[dict, dict, int]] = []
    for records in by_case.values():
        records = sorted(records, key=lambda row: int(row["slice_index"]))
        fluid_records = [record for record in records if record["has_fluid"]]
        empty_records = [record for record in records if not record["has_fluid"]]
        if not fluid_records or not empty_records:
            continue
        for empty in empty_records:
            empty_index = int(empty["slice_index"])
            nearest = min(fluid_records, key=lambda record: abs(int(record["slice_index"]) - empty_index))
            distance = abs(int(nearest["slice_index"]) - empty_index)
            if distance <= args.max_neighbor_distance:
                pairs.append((empty, nearest, distance))

    if len(pairs) < args.num_images:
        raise ValueError(f"Requested {args.num_images} RETOUCH empty-neighbor pairs, found {len(pairs)} within distance {args.max_neighbor_distance}")
    chosen = random.sample(pairs, args.num_images)
    input_dataset = RetouchControlNetDataset([pair[0] for pair in chosen], image_size=256, augment=False)
    mask_dataset = RetouchControlNetDataset([pair[1] for pair in chosen], image_size=256, augment=False)
    images = torch.stack([input_dataset[index]["image"] for index in range(len(chosen))])
    masks = torch.stack([mask_dataset[index]["mask"] for index in range(len(chosen))])
    image_records = []
    mask_records = []
    for source, target, distance in chosen:
        image_records.append(source | {"neighbor_distance": distance, "target_slice_index": target["slice_index"]})
        mask_records.append(target | {"neighbor_distance": distance, "source_slice_index": source["slice_index"]})
    return images, masks, image_records, mask_records


def select_masks(args: argparse.Namespace) -> tuple[torch.Tensor, list[dict]]:
    retouch_records = load_retouch_records(args.retouch_manifest)
    fluid_records = [record for record in retouch_records if record["has_fluid"]]
    if len(fluid_records) < args.num_images:
        raise ValueError(f"Requested {args.num_images} fluid masks, found {len(fluid_records)}")
    chosen = random.sample(fluid_records, args.num_images)
    dataset = RetouchControlNetDataset(chosen, image_size=256, augment=False)
    masks = torch.stack([dataset[index]["mask"] for index in range(len(dataset))])
    return masks, chosen


def load_models(args: argparse.Namespace, device: torch.device):
    network_config = load_json(args.network_config)
    config_ns = argparse.Namespace(**network_config)
    autoencoder = define_instance(config_ns, "autoencoder_def").to(device)
    unet = define_instance(config_ns, "diffusion_unet_def").to(device)
    scheduler = define_instance(config_ns, "noise_scheduler")
    controlnet = build_controlnet(network_config).to(device)

    autoencoder.load_state_dict(torch.load(args.vae_checkpoint, map_location=device, weights_only=True))
    diffusion_ckpt = torch.load(args.diffusion_checkpoint, map_location=device, weights_only=False)
    unet.load_state_dict(diffusion_ckpt["unet_state_dict"])
    controlnet_ckpt = torch.load(args.controlnet_checkpoint, map_location=device, weights_only=False)
    controlnet.load_state_dict(controlnet_ckpt["controlnet_state_dict"])
    scale_factor = diffusion_ckpt["scale_factor"].to(device)

    for model in (autoencoder, unet, controlnet):
        model.eval()
    return autoencoder, unet, controlnet, scheduler, scale_factor


@torch.inference_mode()
def denoise_with_controlnet(
    latents: torch.Tensor,
    masks: torch.Tensor,
    labels: torch.Tensor,
    autoencoder: torch.nn.Module,
    unet: torch.nn.Module,
    controlnet: torch.nn.Module,
    scheduler,
    scale_factor: torch.Tensor,
    timesteps: torch.Tensor,
    next_timesteps: torch.Tensor,
    conditioning_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    image = latents
    for timestep, next_timestep in zip(timesteps, next_timesteps):
        t = torch.full((image.shape[0],), float(timestep), device=image.device)
        down, mid = controlnet(image, timesteps=t, controlnet_cond=masks, class_labels=labels, conditioning_scale=conditioning_scale)
        model_output = unet(image, timesteps=t, class_labels=labels, down_block_additional_residuals=down, mid_block_additional_residual=mid)
        image, _ = scheduler.step(model_output, timestep, image, next_timestep)
    decoded = autoencoder.decode(image / scale_factor).detach().float().cpu().clamp(0, 1)
    return image, decoded


@torch.inference_mode()
def generate_pure(args: argparse.Namespace, masks: torch.Tensor, models: tuple, device: torch.device) -> torch.Tensor:
    autoencoder, unet, controlnet, scheduler, scale_factor = models
    masks = masks.to(device)
    labels = torch.full((masks.shape[0],), args.class_label, dtype=torch.long, device=device)
    latents = torch.randn((masks.shape[0], 4, 64, 64), device=device)
    scheduler.set_timesteps(num_inference_steps=args.num_inference_steps, input_img_size_numel=64 * 64)
    timesteps = scheduler.timesteps
    next_timesteps = torch.cat((timesteps[1:], torch.tensor([0], dtype=timesteps.dtype)))
    _, decoded = denoise_with_controlnet(latents, masks, labels, autoencoder, unet, controlnet, scheduler, scale_factor, timesteps, next_timesteps, args.conditioning_scale)
    return decoded


@torch.inference_mode()
def generate_img2img(args: argparse.Namespace, images: torch.Tensor, masks: torch.Tensor, strength: float, models: tuple, device: torch.device) -> torch.Tensor:
    autoencoder, unet, controlnet, scheduler, scale_factor = models
    images = images.to(device)
    masks = masks.to(device)
    labels = torch.full((images.shape[0],), args.class_label, dtype=torch.long, device=device)
    z_mu, _ = autoencoder.encode(images)
    latents = z_mu * scale_factor

    scheduler.set_timesteps(num_inference_steps=args.num_inference_steps, input_img_size_numel=64 * 64)
    timesteps = scheduler.timesteps
    start_index = min(max(int(round((1.0 - strength) * len(timesteps))), 0), len(timesteps) - 1)
    denoise_timesteps = timesteps[start_index:]
    next_timesteps = torch.cat((denoise_timesteps[1:], torch.tensor([0], dtype=denoise_timesteps.dtype)))
    start_timestep = torch.full((images.shape[0],), float(denoise_timesteps[0]), device=device)
    noisy = scheduler.add_noise(original_samples=latents, noise=torch.randn_like(latents), timesteps=start_timestep)
    _, decoded = denoise_with_controlnet(noisy, masks, labels, autoencoder, unet, controlnet, scheduler, scale_factor, denoise_timesteps, next_timesteps, args.conditioning_scale)
    return decoded


def save_chunked_review_grids(args: argparse.Namespace, originals: torch.Tensor, masks: torch.Tensor, pure: torch.Tensor, edits: dict[float, torch.Tensor]) -> None:
    def as_rgb(tensor: torch.Tensor) -> torch.Tensor:
        tensor = tensor.cpu()
        if tensor.shape[1] == 1:
            return tensor.repeat(1, 3, 1, 1)
        return tensor

    rows = [as_rgb(originals), make_mask_grid(masks), as_rgb(pure)]
    names = ["original", "mask", "pure"]
    for strength in args.strengths:
        rows.append(as_rgb(edits[strength]))
        names.append(f"strength_{strength:g}")
    stacked = torch.cat(rows, dim=0)
    save_image(make_grid(stacked, nrow=args.num_images, padding=2), args.output_dir / "all_cases_rows.png")

    for start in range(0, args.num_images, args.grid_chunk_size):
        end = min(start + args.grid_chunk_size, args.num_images)
        chunk_rows = [tensor[start:end] for tensor in rows]
        chunk = torch.cat(chunk_rows, dim=0)
        save_image(make_grid(chunk, nrow=end - start, padding=2), args.output_dir / f"review_cases_{start:03d}_{end - 1:03d}.png")
    (args.output_dir / "row_order.txt").write_text("\n".join(names) + "\n")


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for OCT ControlNet counterfactual generation.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda")

    if args.input_source == "kermany-normal":
        images, image_records = select_kermany_inputs(args)
        masks, mask_records = select_masks(args)
    else:
        images, masks, image_records, mask_records = select_retouch_empty_neighbor_pairs(args)
    models = load_models(args, device)
    pure = generate_pure(args, masks, models, device)
    edits = {strength: generate_img2img(args, images, masks, strength, models, device) for strength in args.strengths}

    save_image(make_grid(images.cpu(), nrow=args.grid_chunk_size, padding=2), args.output_dir / "inputs_kermany_normal.png")
    save_image(make_grid(make_mask_grid(masks), nrow=args.grid_chunk_size, padding=2), args.output_dir / "target_retouch_masks.png")
    save_image(make_grid(pure.cpu(), nrow=args.grid_chunk_size, padding=2), args.output_dir / "pure_mask_conditioned.png")
    for strength, generated in edits.items():
        save_image(make_grid(generated.cpu(), nrow=args.grid_chunk_size, padding=2), args.output_dir / f"counterfactual_strength_{strength:g}.png")
    save_chunked_review_grids(args, images, masks, pure, edits)

    metrics = {
        "inputs": tensor_stats(images),
        "masks": tensor_stats(masks),
        "pure": tensor_stats(pure),
        "counterfactuals": {str(strength): tensor_stats(generated) for strength, generated in edits.items()},
        "image_records": image_records,
        "mask_records": mask_records,
        "row_order": ["original", "mask", "pure"] + [f"strength_{strength:g}" for strength in args.strengths],
    }
    write_metrics(args.output_dir / "metrics.json", metrics)
    print(json.dumps({key: value for key, value in metrics.items() if key not in {"image_records", "mask_records"}}, indent=2))


if __name__ == "__main__":
    main()
