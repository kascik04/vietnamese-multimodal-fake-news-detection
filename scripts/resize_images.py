#!/usr/bin/env python3
"""
Pre-resize images once to speed up DataLoaders (JPEG decoding of large social-media photos
is the main CPU bottleneck on Kaggle / Colab, which only have 2-4 vCPUs).
Keeps the relative directory structure and file names, so CSV paths stay valid.

Example (Kaggle: inputs are read-only, write to /kaggle/working):
    python scripts/resize_images.py --input_dir /kaggle/input/my-dataset/images \
        --output_dir /kaggle/working/images --max_side 320
    # then train with --image_dir /kaggle/working/images
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from typing import Tuple

from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif")


def _resize_one(job: Tuple[str, str, int, int]) -> str:
    src, dst, max_side, quality = job
    if os.path.exists(dst):
        return "skipped"
    try:
        with Image.open(src) as img:
            img = img.convert("RGB")
            img.thumbnail((max_side, max_side), getattr(Image, "Resampling", Image).BICUBIC)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            # Keep the original file name (extension included) so CSV references still resolve
            if dst.lower().endswith(".png"):
                img.save(dst, format="PNG")
            else:
                img.save(dst, format="JPEG", quality=quality)
        return "ok"
    except Exception as e:
        return f"error: {src}: {e}"


def main():
    parser = argparse.ArgumentParser(description="Pre-resize images for faster training")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_side", type=int, default=320, help="Longest side after resize (>= 224)")
    parser.add_argument("--quality", type=int, default=92)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 2)
    args = parser.parse_args()

    jobs = []
    for root, _, files in os.walk(args.input_dir):
        for name in files:
            if name.lower().endswith(EXTENSIONS):
                src = os.path.join(root, name)
                rel = os.path.relpath(src, args.input_dir)
                jobs.append((src, os.path.join(args.output_dir, rel), args.max_side, args.quality))
    print(f"Resizing {len(jobs)} images -> {args.output_dir} (max side {args.max_side})")

    counts = {"ok": 0, "skipped": 0, "error": 0}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for status in pool.map(_resize_one, jobs, chunksize=32):
            if status.startswith("error"):
                counts["error"] += 1
                print(status)
            else:
                counts[status] += 1
    print(f"Done: {counts}")


if __name__ == "__main__":
    main()
