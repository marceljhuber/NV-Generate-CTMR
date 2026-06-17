# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from monai.data import DataLoader, Dataset
from monai.metrics.fid import FIDMetric
from torchvision.models import Inception_V3_Weights, inception_v3
from torchvision.utils import make_grid, save_image

from .oct_data import define_oct_image_transform, load_oct_manifest
from .train_oct_controlnet_retouch import RetouchControlNetDataset, build_controlnet, make_mask_grid, tensor_stats, write_metrics
from .utils import define_instance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate OCT ControlNet generated samples with FID against RETOUCH and Kermany references.")
    parser.add_argument("--retouch-manifest", type=Path, default=Path("/mnt/nas/media/ubuntu/Thesis/maisi-v2/data/retouch/analysis/retouch_controlnet_spectralis_train_muw_manifest.csv"))
    parser.add_argument("--kermany-root", type=Path, default=Path("/home/user/oct-maisi-cache/OCT"))
    parser.add_argument("--kermany-manifests", type=Path, nargs="+", default=[Path("/mnt/nas/media/ubuntu/Thesis/maisi-v2/data_splits/kermanyv3_oct/train_manifest.csv"), Path("/mnt/nas/media/ubuntu/Thesis/maisi-v2/data_splits/kermanyv3_oct/val_manifest.csv")])
    parser.add_argument("--network-config", type=Path, default=Path("configs/config_network_oct_rflow.json"))
    parser.add_argument("--vae-checkpoint", type=Path, default=Path("models/oct_vae_256_adv_overnight/autoencoder_oct_256_best.pt"))
    parser.add_argument("--diffusion-checkpoint", type=Path, default=Path("models/oct_diffusion_256_adv_overnight/diffusion_oct_256_best.pt"))
    parser.add_argument("--controlnet-checkpoint", type=Path, default=Path("models/oct_controlnet_retouch_256_corrected_png_long_geom_aug_bs32/controlnet_oct_retouch_256_long_geom_aug_best.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/oct_controlnet_fid_geom_best_1000"))
    parser.add_argument("--num-generated", type=int, default=1000)
    parser.add_argument("--num-reference", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--feature-batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-inference-steps", type=int, default=75)
    parser.add_argument("--conditioning-scale", type=float, default=1.0)
    parser.add_argument("--class-label", type=int, default=0)
    parser.add_argument("--save-generated", action="store_true")
    parser.add_argument("--seed", type=int, default=777)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def plain_tensor(value: torch.Tensor) -> torch.Tensor:
    if hasattr(value, "as_tensor"):
        return value.as_tensor()
    return value


def load_retouch_records(manifest: Path) -> list[dict]:
    records = []
    with manifest.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            has_fluid = row.get("has_fluid", "1").lower() in {"1", "true", "yes"}
            records.append(row | {"has_fluid": has_fluid})
    return records


class InceptionFeatures(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        weights = Inception_V3_Weights.IMAGENET1K_V1
        model = inception_v3(weights=weights, aux_logits=True, transform_input=False)
        model.fc = torch.nn.Identity()
        model.eval()
        self.model = model
        self.mean = torch.tensor(weights.transforms().mean).view(1, 3, 1, 1)
        self.std = torch.tensor(weights.transforms().std).view(1, 3, 1, 1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        images = images.float().clamp(0, 1)
        if images.shape[1] == 1:
            images = images.repeat(1, 3, 1, 1)
        images = F.interpolate(images, size=(299, 299), mode="bilinear", align_corners=False)
        return self.model((images - self.mean.to(images.device)) / self.std.to(images.device))


def sample_records(records: list[dict], n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    records = list(records)
    rng.shuffle(records)
    return records[: min(n, len(records))]


def sample_kermany_stratified(records: list[dict], n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    by_label: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_label[record["label"]].append(record)
    labels = [label for label in sorted(by_label) if by_label[label]]
    per_label = max(1, n // max(len(labels), 1))
    chosen = []
    for label in labels:
        label_records = list(by_label[label])
        rng.shuffle(label_records)
        chosen.extend(label_records[:per_label])
    if len(chosen) < n:
        remaining = [record for record in records if record not in chosen]
        rng.shuffle(remaining)
        chosen.extend(remaining[: n - len(chosen)])
    rng.shuffle(chosen)
    return chosen[:n]


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
def generate_controlnet_batch(args: argparse.Namespace, masks: torch.Tensor, models: tuple, device: torch.device) -> torch.Tensor:
    autoencoder, unet, controlnet, scheduler, scale_factor = models
    masks = masks.to(device)
    image = torch.randn((masks.shape[0], 4, 64, 64), device=device)
    labels = torch.full((masks.shape[0],), args.class_label, dtype=torch.long, device=device)
    scheduler.set_timesteps(num_inference_steps=args.num_inference_steps, input_img_size_numel=64 * 64)
    timesteps = scheduler.timesteps
    next_timesteps = torch.cat((timesteps[1:], torch.tensor([0], dtype=timesteps.dtype)))
    for timestep, next_timestep in zip(timesteps, next_timesteps):
        t = torch.full((masks.shape[0],), float(timestep), device=device)
        down, mid = controlnet(image, timesteps=t, controlnet_cond=masks, class_labels=labels, conditioning_scale=args.conditioning_scale)
        model_output = unet(image, timesteps=t, class_labels=labels, down_block_additional_residuals=down, mid_block_additional_residual=mid)
        image, _ = scheduler.step(model_output, timestep, image, next_timestep)
    return autoencoder.decode(image / scale_factor).detach().float().clamp(0, 1)


@torch.inference_mode()
def features_for_tensor_batches(feature_net: torch.nn.Module, batches: list[torch.Tensor], device: torch.device) -> torch.Tensor:
    features = []
    for batch in batches:
        features.append(feature_net(batch.to(device)).detach().cpu())
    return torch.cat(features, dim=0)


@torch.inference_mode()
def features_for_loader(feature_net: torch.nn.Module, loader: DataLoader, device: torch.device, key: str = "image") -> torch.Tensor:
    features = []
    for batch_idx, batch in enumerate(loader):
        images = plain_tensor(batch[key]).to(device, non_blocking=True).contiguous()
        features.append(feature_net(images).detach().cpu())
        if (batch_idx + 1) % 10 == 0:
            print(f"reference features: processed {(batch_idx + 1) * loader.batch_size} images", flush=True)
    return torch.cat(features, dim=0)


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for ControlNet FID evaluation.")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    generated_dir = args.output_dir / "generated"
    if args.save_generated:
        generated_dir.mkdir(parents=True, exist_ok=True)

    retouch_records = load_retouch_records(args.retouch_manifest)
    fluid_records = [record for record in retouch_records if record["has_fluid"]]
    all_records = list(retouch_records)
    selected_mask_records = sample_records(fluid_records, args.num_generated, args.seed)
    mask_dataset = RetouchControlNetDataset(selected_mask_records, image_size=256, augment=False)

    models = load_models(args, device)
    feature_net = InceptionFeatures().to(device).eval()

    generated_features = []
    generated_preview = []
    mask_preview = []
    generated_stats_batches = []
    for start in range(0, len(selected_mask_records), args.batch_size):
        end = min(start + args.batch_size, len(selected_mask_records))
        masks = torch.stack([mask_dataset[index]["mask"] for index in range(start, end)])
        generated = generate_controlnet_batch(args, masks, models, device)
        generated_features.append(feature_net(generated).detach().cpu())
        generated_stats_batches.append(generated.cpu())
        if len(generated_preview) < 32:
            keep = min(generated.shape[0], 32 - len(generated_preview))
            generated_preview.extend(generated[:keep].cpu())
            mask_preview.extend(make_mask_grid(masks[:keep]))
        if args.save_generated:
            for offset, image in enumerate(generated.cpu()):
                save_image(image, generated_dir / f"controlnet_{start + offset:06d}.png")
        if (start // args.batch_size + 1) % 5 == 0 or end == len(selected_mask_records):
            print(f"generated {end}/{len(selected_mask_records)} ControlNet samples", flush=True)

    generated_features_tensor = torch.cat(generated_features, dim=0)
    generated_images_tensor = torch.cat(generated_stats_batches, dim=0)
    save_image(make_grid(torch.stack(generated_preview), nrow=8, padding=2), args.output_dir / "generated_preview_grid.png")
    save_image(make_grid(torch.stack(mask_preview), nrow=8, padding=2), args.output_dir / "mask_preview_grid.png")

    retouch_fluid_dataset = RetouchControlNetDataset(sample_records(fluid_records, args.num_reference, args.seed + 1), image_size=256, augment=False)
    retouch_all_dataset = RetouchControlNetDataset(sample_records(all_records, args.num_reference, args.seed + 2), image_size=256, augment=False)
    retouch_fluid_loader = DataLoader(retouch_fluid_dataset, batch_size=args.feature_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    retouch_all_loader = DataLoader(retouch_all_dataset, batch_size=args.feature_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    kermany_records = []
    for manifest in args.kermany_manifests:
        kermany_records.extend(load_oct_manifest(manifest, args.kermany_root))
    kermany_normal = [record for record in kermany_records if record["label"] == "NORMAL"]
    kermany_transform = define_oct_image_transform(image_size=256, is_train=False, output_dtype=torch.float32, random_aug=False)
    kermany_normal_loader = DataLoader(Dataset(data=sample_records(kermany_normal, args.num_reference, args.seed + 3), transform=kermany_transform), batch_size=args.feature_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    kermany_strat_loader = DataLoader(Dataset(data=sample_kermany_stratified(kermany_records, args.num_reference, args.seed + 4), transform=kermany_transform), batch_size=args.feature_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    references = {
        "retouch_spectralis_fluid": features_for_loader(feature_net, retouch_fluid_loader, device),
        "retouch_spectralis_all": features_for_loader(feature_net, retouch_all_loader, device),
        "kermany_normal_trainval": features_for_loader(feature_net, kermany_normal_loader, device),
        "kermany_stratified_trainval": features_for_loader(feature_net, kermany_strat_loader, device),
    }
    fid_metric = FIDMetric()
    fids = {name: float(fid_metric(generated_features_tensor, features)) for name, features in references.items()}
    summary = {
        "controlnet_checkpoint": str(args.controlnet_checkpoint),
        "vae_checkpoint": str(args.vae_checkpoint),
        "diffusion_checkpoint": str(args.diffusion_checkpoint),
        "retouch_manifest": str(args.retouch_manifest),
        "kermany_manifests": [str(path) for path in args.kermany_manifests],
        "num_generated": int(generated_features_tensor.shape[0]),
        "num_reference_requested": args.num_reference,
        "num_inference_steps": args.num_inference_steps,
        "conditioning_scale": args.conditioning_scale,
        "class_label": args.class_label,
        "generated_stats": tensor_stats(generated_images_tensor),
        "reference_feature_counts": {name: int(features.shape[0]) for name, features in references.items()},
        "fid_generated_vs_reference": fids,
        "feature_model": "torchvision.inception_v3_imagenet_fc_features",
        "note": "FID is an ImageNet-feature sanity metric for comparison, not a retinal disease validity metric.",
    }
    write_metrics(args.output_dir / "controlnet_fid_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
