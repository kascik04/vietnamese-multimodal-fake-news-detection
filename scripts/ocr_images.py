#!/usr/bin/env python3
"""
Offline OCR: read the text printed inside post images and store it in an `ocr_text` column.
===========================================================================================
Screenshot-style fake news (fake official letters, "breaking news" banners, chat captures) carries
its claim inside the image, which a ViT pre-trained on photos barely reads. This script runs OCR
once, offline, and writes the result next to the post text; training then uses it with
`--use_ocr_text` (the encoder sees "<post text>\\nChữ trong ảnh: <ocr text>").

    pip install easyocr          # Kaggle / Colab; uses the GPU when available
    python scripts/ocr_images.py --csv data/train.csv data/val.csv data/test.csv --image_dir data/images
    python -m vmtf_net.train ... --use_ocr_text

- Up to 3 images per post (image_1..3_local_path) are read; their texts are joined with " | ".
- Results are cached per image file (--cache, default <image_dir>/ocr_cache.json), so the script can
  be interrupted / re-run and the same image is never read twice (train/val/test, K-fold copies).
- Low-confidence fragments and near-empty results are dropped (photos give OCR garbage).
- The CSVs are updated in place (only the `ocr_text` column changes) unless --output_dir is given.
  Run it AFTER validate_and_preprocess_dataset.py, or before: that script keeps `ocr_text`.
- When reporting results, treat --use_ocr_text as its own ablation row.
"""

import argparse
import json
import os
import re
import sys
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

OCR_TEXT_COL = "ocr_text"
IMAGE_COLS = ["image_1_local_path", "image_2_local_path", "image_3_local_path"]
# One OCR engine = a function: image path -> [(text fragment, confidence in [0, 1]), ...]
OcrEngine = Callable[[str], List[Tuple[str, float]]]


def build_easyocr_engine(langs: Sequence[str], gpu: Optional[bool] = None) -> OcrEngine:
    """EasyOCR reader (supports Vietnamese). Models are downloaded on first use (~100 MB)."""
    try:
        import easyocr
    except ImportError as e:
        raise SystemExit("easyocr is not installed. Run: pip install easyocr") from e
    if gpu is None:
        try:
            import torch

            gpu = torch.cuda.is_available()
        except ImportError:
            gpu = False
    reader = easyocr.Reader(list(langs), gpu=gpu)

    def _read(path: str) -> List[Tuple[str, float]]:
        return [(str(text), float(conf)) for _, text, conf in reader.readtext(path)]

    return _read


def clean_ocr(fragments: List[Tuple[str, float]], min_conf: float, min_words: int, max_chars: int) -> str:
    """Keep confident fragments, normalize spaces, drop near-empty results, cap the length."""
    kept = [re.sub(r"\s+", " ", t).strip() for t, conf in fragments if conf >= min_conf]
    text = " ".join(t for t in kept if len(t) >= 2)
    if len(re.findall(r"\w+", text)) < min_words:
        return ""
    return text[:max_chars].strip()


def resolve_image(image_dir: str, value) -> Optional[str]:
    if value is None or pd.isna(value) or str(value).strip() == "":
        return None
    v = str(value).strip()
    for candidate in (v, os.path.join(image_dir, v), os.path.join(image_dir, os.path.basename(v))):
        if os.path.isfile(candidate):
            return os.path.abspath(candidate)
    return None


def run_ocr(
    csv_paths: Sequence[str],
    image_dir: str,
    engine: OcrEngine,
    cache_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    min_conf: float = 0.3,
    min_words: int = 3,
    max_chars: int = 600,
    save_every: int = 50,
) -> Dict[str, int]:
    """Add / refresh the `ocr_text` column of every CSV. Returns counters."""
    cache_path = cache_path or os.path.join(image_dir, "ocr_cache.json")
    cache: Dict[str, str] = {}
    if os.path.isfile(cache_path):
        with open(cache_path, encoding="utf-8") as f:
            cache = json.load(f)

    def _save_cache() -> None:
        os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
        tmp = cache_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False)
        os.replace(tmp, cache_path)

    image_root = os.path.abspath(image_dir)
    stats = {"posts": 0, "posts_with_ocr_text": 0, "images_read": 0, "images_from_cache": 0, "errors": 0}
    for csv_path in csv_paths:
        df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
        cols = [c for c in IMAGE_COLS if c in df.columns]
        texts: List[str] = []
        for values in zip(*(df[c].tolist() for c in cols)) if cols else [()] * len(df):
            parts: List[str] = []
            for value in values:
                path = resolve_image(image_dir, value)
                if path is None:
                    continue
                # cache key: path relative to image_dir (stable across machines), else the file name
                key = os.path.relpath(path, image_root).replace(os.sep, "/") if path.startswith(image_root) \
                    else os.path.basename(path)
                if key in cache:
                    stats["images_from_cache"] += 1
                else:
                    try:
                        cache[key] = clean_ocr(engine(path), min_conf, min_words, max_chars)
                    except Exception as e:  # unreadable image: do not stop a long run
                        print(f"[warning] OCR failed for {path}: {e}")
                        cache[key] = ""
                        stats["errors"] += 1
                    stats["images_read"] += 1
                    if stats["images_read"] % save_every == 0:
                        _save_cache()
                        print(f"  ... {stats['images_read']} images read")
                if cache[key] and cache[key] not in parts:
                    parts.append(cache[key])
            texts.append(" | ".join(parts)[: max_chars * 2])
        df[OCR_TEXT_COL] = texts
        stats["posts"] += len(df)
        stats["posts_with_ocr_text"] += sum(bool(t) for t in texts)

        out_path = csv_path
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            out_path = os.path.join(output_dir, os.path.basename(csv_path))
        df.to_csv(out_path, index=False, encoding="utf-8")
        print(f"{out_path}: {sum(bool(t) for t in texts)}/{len(df)} posts have text in their images")
    _save_cache()
    return stats


def main():
    parser = argparse.ArgumentParser(description="OCR post images into an `ocr_text` column")
    parser.add_argument("--csv", nargs="+", required=True, help="CSV file(s): train/val/test or the full dataset")
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--output_dir", default=None, help="Write updated CSVs here (default: in place)")
    parser.add_argument("--cache", default=None, help="OCR cache JSON (default: <image_dir>/ocr_cache.json)")
    parser.add_argument("--langs", nargs="+", default=["vi", "en"])
    parser.add_argument("--min_conf", type=float, default=0.3, help="Drop fragments below this confidence")
    parser.add_argument("--min_words", type=int, default=3, help="Ignore images with fewer words (logos, noise)")
    parser.add_argument("--max_chars", type=int, default=600, help="Max OCR characters kept per image")
    parser.add_argument("--cpu", action="store_true", help="Force CPU (slow)")
    args = parser.parse_args()

    engine = build_easyocr_engine(args.langs, gpu=False if args.cpu else None)
    stats = run_ocr(args.csv, args.image_dir, engine, args.cache, args.output_dir,
                    args.min_conf, args.min_words, args.max_chars)
    print(f"Done: {stats}")
    print("Train with:  python -m vmtf_net.train ... --use_ocr_text")


if __name__ == "__main__":
    main()
