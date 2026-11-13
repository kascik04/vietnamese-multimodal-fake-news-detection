#!/usr/bin/env python3
"""
Package a split dataset into one portable zip (same layout as vmtf_sample_dataset.zip):

    dataset.zip
    ├── train.csv
    ├── val.csv
    ├── test.csv
    ├── folds/fold_*/{train,val,test}.csv     (optional, --include_folds)
    └── images/<file>                         (flat; only images referenced by the CSVs)

image_k_local_path columns are rewritten to the flat file name, so after unzipping anywhere:
    python -m vmtf_net.train --train_csv <dir>/train.csv ... --image_dir <dir>/images
This is also the `data.zip` expected by notebooks/colab_a100_drive.ipynb.

Example (after validate_and_preprocess_dataset.py wrote data/train.csv, val.csv, test.csv):
    python scripts/package_dataset.py --split_dir data --image_dir ../data-collector/data/export/images \
        --output vmtf_dataset.zip --include_folds
"""

import argparse
import io
import os
import re
import sys
import zipfile
from typing import Dict, List, Optional

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

SPLITS = ("train", "val", "test")
IMAGE_SLOTS = (1, 2, 3)


def resolve_image(image_dir: str, value) -> Optional[str]:
    """Find the image file referenced by a CSV cell (relative path, bare name or absolute path)."""
    if value is None or pd.isna(value) or str(value).strip() == "":
        return None
    v = str(value).strip()
    for candidate in (v, os.path.join(image_dir, v), os.path.join(image_dir, os.path.basename(v))):
        if os.path.isfile(candidate):
            return os.path.abspath(candidate)
    return None


def flat_name(sample_id: str, slot: int, src: str, used: Dict[str, str]) -> str:
    """Unique flat file name '<sample_id>_<slot>.<ext>'."""
    ext = os.path.splitext(src)[1].lower() or ".jpg"
    base = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(sample_id))[:80].strip("_") or "img"
    name, k = f"{base}_{slot}{ext}", 1
    while name in used.values():
        k += 1
        name = f"{base}_{slot}_{k}{ext}"
    return name


def collect_csvs(split_dir: str, include_folds: bool) -> Dict[str, str]:
    """Archive path -> CSV path on disk."""
    found = {}
    for s in SPLITS:
        p = os.path.join(split_dir, f"{s}.csv")
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Missing {p}: run validate_and_preprocess_dataset.py first.")
        found[f"{s}.csv"] = p
    if include_folds:
        folds_dir = os.path.join(split_dir, "folds")
        if os.path.isdir(folds_dir):
            for fold in sorted(os.listdir(folds_dir)):
                for s in SPLITS:
                    p = os.path.join(folds_dir, fold, f"{s}.csv")
                    if os.path.isfile(p):
                        found[f"folds/{fold}/{s}.csv"] = p
    return found


def package(split_dir: str, image_dir: str, output: str, include_folds: bool = False) -> Dict[str, int]:
    csvs = collect_csvs(split_dir, include_folds)
    src_to_name: Dict[str, str] = {}   # absolute source path -> flat name (shared across splits)
    stats = {"rows": 0, "images": 0, "missing_images": 0}
    frames: Dict[str, pd.DataFrame] = {}

    for arc, path in csvs.items():
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
        for slot in IMAGE_SLOTS:
            col = f"image_{slot}_local_path"
            if col not in df.columns:
                continue
            new_values: List[str] = []
            for sid, value in zip(df.get("sample_id", pd.Series([f"row{i}" for i in range(len(df))])), df[col]):
                src = resolve_image(image_dir, value)
                if src is None:
                    if str(value).strip():
                        stats["missing_images"] += 1
                    new_values.append("")  # never ship a dangling path
                    continue
                if src not in src_to_name:
                    src_to_name[src] = flat_name(sid, slot, src, src_to_name)
                new_values.append(src_to_name[src])
            df[col] = new_values
        frames[arc] = df
        if "/" not in arc:
            stats["rows"] += len(df)

    os.makedirs(os.path.dirname(os.path.abspath(output)) or ".", exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for arc, df in frames.items():
            buf = io.StringIO()
            df.to_csv(buf, index=False)
            zf.writestr(arc, buf.getvalue().encode("utf-8"))
        for src, name in src_to_name.items():
            zf.write(src, f"images/{name}", compress_type=zipfile.ZIP_STORED)  # JPEGs are already compressed
    stats["images"] = len(src_to_name)
    return stats


def main():
    parser = argparse.ArgumentParser(description="Package train/val/test CSVs + images into one zip")
    parser.add_argument("--split_dir", default="data", help="Directory with train.csv, val.csv, test.csv")
    parser.add_argument("--image_dir", required=True, help="Directory the image_*_local_path values are relative to")
    parser.add_argument("--output", default="vmtf_dataset.zip")
    parser.add_argument("--include_folds", action="store_true", help="Also pack <split_dir>/folds/fold_*/")
    args = parser.parse_args()

    stats = package(args.split_dir, args.image_dir, args.output, args.include_folds)
    size_mb = os.path.getsize(args.output) / 1024 ** 2
    print(f"Saved {args.output} ({size_mb:.1f} MB): {stats['rows']} rows in train/val/test, "
          f"{stats['images']} images, {stats['missing_images']} missing image references dropped.")
    if stats["missing_images"]:
        print("  -> check --image_dir: some image paths in the CSVs do not exist on disk.")


if __name__ == "__main__":
    main()
