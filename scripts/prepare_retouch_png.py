#!/usr/bin/env python
"""Convert RETOUCH MetaImage OCT volumes to full-resolution PNG slices."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from skimage.io import imsave
from tqdm import tqdm


LABELS = {
    1: "irf",
    2: "srf",
    3: "ped",
}


def parse_mhd(path: Path) -> dict[str, str]:
    header: dict[str, str] = {}
    with path.open() as f:
        for line in f:
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            header[key.strip()] = value.strip()
    return header


def dtype_from_mhd(element_type: str) -> np.dtype:
    dtypes = {
        "MET_UCHAR": np.dtype("uint8"),
        "MET_CHAR": np.dtype("int8"),
        "MET_USHORT": np.dtype("uint16"),
        "MET_SHORT": np.dtype("int16"),
        "MET_UINT": np.dtype("uint32"),
        "MET_INT": np.dtype("int32"),
        "MET_FLOAT": np.dtype("float32"),
        "MET_DOUBLE": np.dtype("float64"),
    }
    try:
        return dtypes[element_type]
    except KeyError as exc:
        raise ValueError(f"Unsupported ElementType {element_type!r}") from exc


def read_mhd_volume(path: Path) -> tuple[np.ndarray, dict[str, str]]:
    header = parse_mhd(path)
    dim_size = [int(value) for value in header["DimSize"].split()]
    dtype = dtype_from_mhd(header["ElementType"])
    raw_path = path.parent / header["ElementDataFile"]
    volume = np.fromfile(raw_path, dtype=dtype)
    expected = int(np.prod(dim_size))
    if volume.size != expected:
        raise ValueError(f"{raw_path} has {volume.size} values, expected {expected} from DimSize={dim_size}")

    if header.get("BinaryDataByteOrderMSB", "False") == "True":
        volume = volume.byteswap().newbyteorder()

    # RETOUCH stores dimensions as axial depth, lateral width, B-scan count.
    # Transposing each raw slice below restores conventional B-scan layout.
    x_depth, y_width, z_slices = dim_size
    return volume.reshape((z_slices, y_width, x_depth)), header


def infer_scanner(header: dict[str, str]) -> str:
    x_depth, y_width, z_slices = [int(value) for value in header["DimSize"].split()]
    if (x_depth, y_width, z_slices) == (512, 1024, 128):
        return "Cirrus"
    if z_slices == 49:
        return "Spectralis"
    return "Topcon"


def split_name(case_dir: Path, retouch_root: Path) -> tuple[str, str | None]:
    relative = case_dir.relative_to(retouch_root)
    parts = relative.parts
    if parts[0] == "TrainingSet-Release":
        return "train", parts[1]
    if parts[0] == "TestSet-MUW":
        return "test_muw", None
    if parts[0] == "TestSet-Radboud":
        return "test_radboud", None
    raise ValueError(f"Unknown RETOUCH subset for {case_dir}")


def save_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    imsave(path, image, check_contrast=False)


def convert_case(case_dir: Path, retouch_root: Path, output_root: Path) -> tuple[list[dict[str, str | int | float]], dict[str, str | int]]:
    oct_volume, oct_header = read_mhd_volume(case_dir / "oct.mhd")
    ref_volume, ref_header = read_mhd_volume(case_dir / "reference.mhd")
    if oct_volume.shape != ref_volume.shape:
        raise ValueError(f"OCT/reference shape mismatch in {case_dir}: {oct_volume.shape} vs {ref_volume.shape}")

    split, scanner_from_path = split_name(case_dir, retouch_root)
    scanner = scanner_from_path or infer_scanner(oct_header)
    case_id = case_dir.name
    dim_size = [int(value) for value in oct_header["DimSize"].split()]
    spacing = [float(value) for value in oct_header.get("ElementSpacing", "nan nan nan").split()]
    rows: list[dict[str, str | int | float]] = []
    unique_labels: set[int] = set()

    for index in range(oct_volume.shape[0]):
        oct_slice = oct_volume[index].T
        mask_slice = ref_volume[index].T
        unique_labels.update(int(value) for value in np.unique(mask_slice))

        stem = f"slice_{index:03d}.png"
        image_path = output_root / "images" / scanner / split / case_id / stem
        mask_path = output_root / "masks_label" / scanner / split / case_id / stem
        save_png(image_path, oct_slice)
        save_png(mask_path, mask_slice.astype(np.uint8))

        binary_paths: dict[str, str] = {}
        for label_value, label_name in LABELS.items():
            binary_path = output_root / f"masks_{label_name}" / scanner / split / case_id / stem
            save_png(binary_path, ((mask_slice == label_value) * 255).astype(np.uint8))
            binary_paths[f"{label_name}_mask_path"] = str(binary_path)

        rows.append(
            {
                "split": split,
                "scanner": scanner,
                "case_id": case_id,
                "slice_index": index,
                "image_path": str(image_path),
                "label_mask_path": str(mask_path),
                **binary_paths,
                "height": int(oct_slice.shape[0]),
                "width": int(oct_slice.shape[1]),
                "dim_x_depth": dim_size[0],
                "dim_y_width": dim_size[1],
                "dim_z_slices": dim_size[2],
                "spacing_x": spacing[0],
                "spacing_y": spacing[1],
                "spacing_z": spacing[2],
                "raw_case_dir": str(case_dir),
            }
        )

    volume_row = {
        "split": split,
        "scanner": scanner,
        "case_id": case_id,
        "num_slices": int(oct_volume.shape[0]),
        "height": int(oct_volume.shape[2]),
        "width": int(oct_volume.shape[1]),
        "dim_x_depth": dim_size[0],
        "dim_y_width": dim_size[1],
        "dim_z_slices": dim_size[2],
        "spacing_x": spacing[0],
        "spacing_y": spacing[1],
        "spacing_z": spacing[2],
        "scan_position": oct_header.get("ScanPosition", ""),
        "unique_label_values": " ".join(str(value) for value in sorted(unique_labels)),
        "raw_case_dir": str(case_dir),
        "reference_element_type": ref_header["ElementType"],
    }
    return rows, volume_row


def discover_cases(retouch_root: Path) -> list[Path]:
    return sorted(path.parent for path in retouch_root.rglob("oct.mhd") if (path.parent / "reference.mhd").exists())


def write_csv(path: Path, rows: list[dict[str, str | int | float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write to {path}")
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retouch-root", type=Path, required=True, help="Extracted RETOUCH directory containing TrainingSet-Release")
    parser.add_argument("--output-root", type=Path, required=True, help="Directory for PNG images, masks, and manifests")
    parser.add_argument("--limit-cases", type=int, default=None, help="Optional case limit for smoke testing")
    args = parser.parse_args()

    cases = discover_cases(args.retouch_root)
    if args.limit_cases is not None:
        cases = cases[: args.limit_cases]
    if not cases:
        raise ValueError(f"No RETOUCH cases found under {args.retouch_root}")

    slice_rows: list[dict[str, str | int | float]] = []
    volume_rows: list[dict[str, str | int]] = []
    for case_dir in tqdm(cases, desc="Converting RETOUCH cases"):
        rows, volume_row = convert_case(case_dir, args.retouch_root, args.output_root)
        slice_rows.extend(rows)
        volume_rows.append(volume_row)

    write_csv(args.output_root / "slice_manifest.csv", slice_rows)
    write_csv(args.output_root / "volume_manifest.csv", volume_rows)
    print(f"Converted {len(volume_rows)} volumes and {len(slice_rows)} B-scans to {args.output_root}")


if __name__ == "__main__":
    main()
