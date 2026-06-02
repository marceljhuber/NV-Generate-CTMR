# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");

from __future__ import annotations

import csv
from pathlib import Path

import torch
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    Lambdad,
    LoadImaged,
    RandAdjustContrastd,
    RandFlipd,
    RandGaussianNoised,
    RandRotated,
    RandScaleIntensityd,
    RandShiftIntensityd,
    Resized,
    ScaleIntensityd,
    SelectItemsd,
)


OCT_LABEL_TO_ID = {
    "unknown": 0,
    "CNV": 1,
    "DME": 2,
    "DRUSEN": 3,
    "NORMAL": 4,
}
OCT_ID_TO_LABEL = {value: key for key, value in OCT_LABEL_TO_ID.items()}


def ensure_single_oct_channel(image: torch.Tensor) -> torch.Tensor:
    """Return one grayscale OCT channel for grayscale or RGB-loaded images."""
    if image.ndim == 4 and image.shape[-1] in (3, 4):
        image = image[..., 0]
    if image.shape[0] > 1:
        image = image[:1]
    return image


def load_oct_manifest(manifest_path: str | Path, dataset_root: str | Path) -> list[dict]:
    """Load a KermanyV3 OCT CSV manifest into MONAI dict records."""
    manifest_path = Path(manifest_path)
    dataset_root = Path(dataset_root)
    records: list[dict] = []

    with manifest_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"relative_path", "label", "patient_id", "image_index", "source_split"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Manifest {manifest_path} is missing columns: {sorted(missing)}")

        for row in reader:
            label = row["label"]
            if label not in OCT_LABEL_TO_ID:
                raise ValueError(f"Unknown OCT label {label!r} in {manifest_path}")

            records.append(
                {
                    "image": str(dataset_root / row["relative_path"]),
                    "relative_path": row["relative_path"],
                    "label": label,
                    "class_label": OCT_LABEL_TO_ID[label],
                    "patient_id": row["patient_id"],
                    "image_index": int(row["image_index"]),
                    "source_split": row["source_split"],
                }
            )

    return records


def define_oct_image_transform(
    image_size: int | tuple[int, int] = 128,
    is_train: bool = True,
    output_dtype: torch.dtype = torch.float32,
    random_aug: bool = True,
) -> Compose:
    """Define 2D OCT transforms for local JPEG/PNG images."""
    spatial_size = (image_size, image_size) if isinstance(image_size, int) else tuple(image_size)

    transforms = [
        SelectItemsd(keys=["image", "class_label", "label", "patient_id", "image_index", "source_split", "relative_path"], allow_missing_keys=True),
        LoadImaged(keys="image", image_only=True),
        EnsureChannelFirstd(keys="image", channel_dim="no_channel"),
        Lambdad(keys="image", func=ensure_single_oct_channel),
        Lambdad(keys="image", func=lambda x: torch.rot90(x, k=-1, dims=(-2, -1))),
        EnsureTyped(keys="image", dtype=torch.float32),
        ScaleIntensityd(keys="image", minv=0.0, maxv=1.0),
        Resized(keys="image", spatial_size=spatial_size, mode="bilinear"),
    ]

    if is_train and random_aug:
        transforms.extend(
            [
                RandFlipd(keys="image", prob=0.5, spatial_axis=1),
                RandRotated(keys="image", prob=0.2, range_x=0.08, mode="bilinear", padding_mode="border"),
                RandAdjustContrastd(keys="image", prob=0.2, gamma=(0.8, 1.25)),
                RandScaleIntensityd(keys="image", prob=0.2, factors=0.05),
                RandShiftIntensityd(keys="image", prob=0.2, offsets=0.03),
                RandGaussianNoised(keys="image", prob=0.1, mean=0.0, std=0.01),
            ]
        )

    transforms.extend(
        [
            EnsureTyped(keys="image", dtype=output_dtype),
            EnsureTyped(keys="class_label", dtype=torch.long),
        ]
    )
    return Compose(transforms)
