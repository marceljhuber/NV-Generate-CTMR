# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from monai.data import DataLoader, Dataset
from monai.metrics.fid import FIDMetric
from monai.networks.schedulers import RFlowScheduler
from torchvision.models import Inception_V3_Weights, inception_v3
from torchvision.utils import make_grid, save_image

from .oct_data import OCT_LABEL_TO_ID, OCT_ID_TO_LABEL, define_oct_image_transform, load_oct_manifest
from .train_oct_diffusion import load_latent_split
from .utils import define_instance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate OCT images from diffusion and compute FID against real OCT images.")
    parser.add_argument("--dataset-root", type=Path, default=Path("/home/user/oct-maisi-cache/OCT"))
    parser.add_argument("--manifests", type=Path, nargs="+", default=[Path("../../data_splits/kermanyv3_oct/train_manifest.csv"), Path("../../data_splits/kermanyv3_oct/val_manifest.csv")])
    parser.add_argument("--latents-dir", type=Path, default=Path("outputs/oct_latents_128_overnight"))
    parser.add_argument("--vae-checkpoint", type=Path, default=Path("models/oct_vae_128_overnight/autoencoder_oct_128_best.pt"))
    parser.add_argument("--diffusion-checkpoint", type=Path, default=Path("models/oct_diffusion_128_overnight/diffusion_oct_128_best.pt"))
    parser.add_argument("--network-config", type=Path, default=Path("configs/config_network_oct_rflow.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/oct_diffusion_128_overnight/fid_generated_10pct"))
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--sample-percent", type=float, default=10.0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--num-inference-steps", type=int, default=30)
    parser.add_argument("--cfg-guidance-scale", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--generation-mode",
        choices=["proportional", "balanced", "unconditional", "single-class"],
        default="proportional",
        help="How to allocate generated images across classes. 'unconditional' uses class 0 for every generated image.",
    )
    parser.add_argument(
        "--single-class",
        choices=sorted(OCT_LABEL_TO_ID),
        default=None,
        help="If --generation-mode single-class, generate only images for this class label.",
    )
    parser.add_argument("--save-generated", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def plain_tensor(value: torch.Tensor) -> torch.Tensor:
    if hasattr(value, "as_tensor"):
        return value.as_tensor()
    return value


def count_by_class(records: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for record in records:
        counts[record["label"]] += 1
    return dict(sorted(counts.items()))


def stratified_sample(records: list[dict], counts: dict[str, int], seed: int) -> list[dict]:
    rng = random.Random(seed)
    by_label: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_label[record["label"]].append(record)

    sampled = []
    for label, n in sorted(counts.items()):
        class_records = list(by_label[label])
        rng.shuffle(class_records)
        sampled.extend(class_records[: min(n, len(class_records))])
    rng.shuffle(sampled)
    return sampled


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
        mean = self.mean.to(images.device)
        std = self.std.to(images.device)
        return self.model((images - mean) / std)


@torch.inference_mode()
def generate_batch(
    unet: torch.nn.Module,
    autoencoder: torch.nn.Module,
    scheduler: RFlowScheduler,
    latent_shape: tuple[int, int, int],
    scale_factor: torch.Tensor,
    device: torch.device,
    class_id: int,
    batch_size: int,
    num_inference_steps: int,
    cfg_guidance_scale: float,
) -> torch.Tensor:
    image = torch.randn((batch_size, *latent_shape), device=device)
    labels = torch.full((batch_size,), class_id, dtype=torch.long, device=device)
    scheduler.set_timesteps(num_inference_steps=num_inference_steps, input_img_size_numel=torch.prod(torch.tensor(latent_shape[1:])))
    timesteps = scheduler.timesteps
    next_timesteps = torch.cat((timesteps[1:], torch.tensor([0], dtype=timesteps.dtype)))

    for timestep, next_timestep in zip(timesteps, next_timesteps):
        t = torch.full((batch_size,), float(timestep), device=device)
        if cfg_guidance_scale > 0:
            model_input = torch.cat([image, image], dim=0)
            model_t = torch.cat([t, t], dim=0)
            model_labels = torch.cat([labels, torch.zeros_like(labels)], dim=0)
            pred_cond, pred_uncond = unet(model_input, timesteps=model_t, class_labels=model_labels).chunk(2)
            model_output = pred_uncond + cfg_guidance_scale * (pred_cond - pred_uncond)
        else:
            model_output = unet(image, timesteps=t, class_labels=labels)
        image, _ = scheduler.step(model_output, timestep, image, next_timestep)
    return autoencoder.decode(image / scale_factor).detach().float().clamp(0, 1)


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for OCT diffusion FID evaluation.")
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    generated_dir = args.output_dir / "generated"
    if args.save_generated:
        generated_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for manifest_path in args.manifests:
        records.extend(load_oct_manifest(manifest_path, args.dataset_root))
    real_counts = count_by_class(records)
    proportional_counts = {label: max(1, round(count * args.sample_percent / 100.0)) for label, count in real_counts.items()}

    if args.generation_mode == "unconditional":
        total_target = sum(proportional_counts.values())
        synth_counts = {"unknown": total_target}
    elif args.generation_mode == "balanced":
        per_class = max(1, round(sum(real_counts.values()) * args.sample_percent / 100.0 / max(len(real_counts), 1)))
        synth_counts = {label: per_class for label in sorted(real_counts)}
    elif args.generation_mode == "single-class":
        if args.single_class is None:
            raise ValueError("--single-class is required for --generation-mode single-class.")
        target_for_class = max(1, round(real_counts.get(args.single_class, 0) * args.sample_percent / 100.0))
        synth_counts = {args.single_class: target_for_class}
    else:
        synth_counts = proportional_counts

    real_sampled_records = stratified_sample(records, proportional_counts, args.seed)

    transform = define_oct_image_transform(image_size=args.image_size, is_train=False, output_dtype=torch.float32, random_aug=False)
    dataset = Dataset(data=real_sampled_records, transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    network_config = load_json(args.network_config)
    config_ns = argparse.Namespace(**network_config)
    unet = define_instance(config_ns, "diffusion_unet_def").to(device)
    autoencoder = define_instance(config_ns, "autoencoder_def").to(device)
    scheduler = define_instance(config_ns, "noise_scheduler")
    autoencoder.load_state_dict(torch.load(args.vae_checkpoint, map_location=device, weights_only=True))
    ckpt = torch.load(args.diffusion_checkpoint, map_location=device, weights_only=True)
    unet.load_state_dict(ckpt["unet_state_dict"])
    scale_factor = ckpt["scale_factor"].to(device)
    unet.eval()
    autoencoder.eval()
    feature_net = InceptionFeatures().to(device).eval()

    train_latents, _ = load_latent_split(args.latents_dir, "train")
    latent_shape = tuple(train_latents.shape[1:])

    real_features = []
    for batch_idx, batch in enumerate(loader):
        images = plain_tensor(batch["image"]).to(device, non_blocking=True).contiguous()
        real_features.append(feature_net(images).detach().cpu())
        if (batch_idx + 1) % 10 == 0:
            print(f"real features: processed {(batch_idx + 1) * args.batch_size} images", flush=True)

    generated_features = []
    preview_images = []
    generated_counts: dict[str, int] = {}
    for label, count in sorted(synth_counts.items()):
        class_id = OCT_LABEL_TO_ID[label]
        generated_counts[label] = 0
        num_batches = math.ceil(count / args.batch_size)
        for batch_idx in range(num_batches):
            current_batch = min(args.batch_size, count - batch_idx * args.batch_size)
            images = generate_batch(
                unet,
                autoencoder,
                scheduler,
                latent_shape,
                scale_factor,
                device,
                class_id,
                current_batch,
                args.num_inference_steps,
                args.cfg_guidance_scale,
            )
            generated_features.append(feature_net(images).detach().cpu())
            generated_counts[label] += current_batch
            if len(preview_images) < 32:
                preview_images.append(images[: max(0, min(images.shape[0], 32 - len(preview_images)))].cpu())
            if args.save_generated:
                class_dir = generated_dir / label
                class_dir.mkdir(parents=True, exist_ok=True)
                start = batch_idx * args.batch_size
                for idx, image in enumerate(images.cpu()):
                    save_image(image, class_dir / f"{label.lower()}_{start + idx:06d}.png")
            if (batch_idx + 1) % 10 == 0 or batch_idx + 1 == num_batches:
                print(f"generated {generated_counts[label]}/{count} images for {label}", flush=True)

    if preview_images:
        preview = torch.cat(preview_images, dim=0)
        save_image(make_grid(preview, nrow=8, padding=2), args.output_dir / "generated_preview_grid.png")

    real_features_tensor = torch.cat(real_features, dim=0)
    generated_features_tensor = torch.cat(generated_features, dim=0)
    fid = float(FIDMetric()(generated_features_tensor, real_features_tensor))
    summary = {
        "manifests": [str(path) for path in args.manifests],
        "dataset_root": str(args.dataset_root),
        "vae_checkpoint": str(args.vae_checkpoint),
        "diffusion_checkpoint": str(args.diffusion_checkpoint),
        "sample_percent": args.sample_percent,
        "num_inference_steps": args.num_inference_steps,
        "cfg_guidance_scale": args.cfg_guidance_scale,
        "generation_mode": args.generation_mode,
        "single_class": args.single_class,
        "real_counts_full": real_counts,
        "target_counts_by_class": synth_counts,
        "generated_counts_by_class": generated_counts,
        "num_real_images_for_fid": int(real_features_tensor.shape[0]),
        "num_generated_images_for_fid": int(generated_features_tensor.shape[0]),
        "feature_model": "torchvision.inception_v3_imagenet_fc_features",
        "fid_generated_vs_real": fid,
    }
    out_path = args.output_dir / f"diffusion_generated_fid_{args.sample_percent:g}pct.json"
    with out_path.open("w") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
