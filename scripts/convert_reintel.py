#!/usr/bin/env python3
"""
Convert ReINTEL (VLSP 2020) files to the VMTF-Net 55-column schema.
===================================================================
ReINTEL columns (names are auto-detected, override with --*_col if your copy differs):
    id, user_name, post_message, timestamp_post, num_like_post, num_comment_post,
    num_share_post, image (file name / URL / list, optional), label (1 = unreliable, 0 = reliable)

Images are matched, in order, by:
  1. the image column (file name, relative path, URL basename, or a list of them),
  2. files in --image_dir named after the post id: <id>.jpg, <id>_0.png, <id>-1.jpg ...
Missing images are fine: the model handles them with has_image = 0.

Social counts are parsed robustly ("1,2K", "3.4k", "2 tr", "" ...). Unparseable counts stay
empty and are reported (validate_and_preprocess_dataset.py later fills them with 0).

Example:
    python scripts/convert_reintel.py \
        --input /kaggle/input/reintel/public_train.csv \
        --image_dir /kaggle/input/reintel/images \
        --output data/reintel_vmtf.csv

    python scripts/validate_and_preprocess_dataset.py \
        --input_csv data/reintel_vmtf.csv --image_dir /kaggle/input/reintel/images \
        --output_dir data --split_by group          # or --split_by source / time, --kfold 5
"""

import argparse
import ast
import json
import logging
import os
import re
import sys
from typing import Dict, List, Optional
from urllib.parse import urlparse

import numpy as np
import pandas as pd

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPTS_DIR)

from validate_and_preprocess_dataset import STANDARD_COLUMNS  # noqa: E402

logger = logging.getLogger("ReINTELConverter")

CANDIDATES: Dict[str, List[str]] = {
    "id": ["id", "post_id", "ID"],
    "user": ["user_name", "uid", "user_id", "username", "page_name"],
    "text": ["post_message", "message", "text", "content"],
    "timestamp": ["timestamp_post", "timestamp", "created_time", "time"],
    "like": ["num_like_post", "like", "likes", "num_like"],
    "comment": ["num_comment_post", "comment", "comments", "num_comment"],
    "share": ["num_share_post", "share", "shares", "num_share"],
    "image": ["image", "images", "image_path", "img", "image_url", "image_urls"],
    "label": ["label", "Label", "target"],
}
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp")


# ----------------------------------------------------------------------
# Parsing helpers
# ----------------------------------------------------------------------
def read_any(path: str) -> pd.DataFrame:
    """Read CSV / TSV / JSON (records) / JSONL / Excel."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".json",):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):  # {"data": [...]} or {id: record}
            data = data.get("data", list(data.values()))
        return pd.DataFrame(data)
    if ext in (".jsonl", ".ndjson"):
        return pd.read_json(path, lines=True)
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(path)
    sep = "\t" if ext == ".tsv" else ","
    return pd.read_csv(path, sep=sep, dtype=str, keep_default_na=False, na_values=[""])


def resolve_columns(df: pd.DataFrame, overrides: Dict[str, Optional[str]]) -> Dict[str, Optional[str]]:
    cols = {}
    for key, cands in CANDIDATES.items():
        if overrides.get(key):
            if overrides[key] not in df.columns:
                raise KeyError(f"--{key}_col '{overrides[key]}' not in columns {list(df.columns)}")
            cols[key] = overrides[key]
        else:
            cols[key] = next((c for c in cands if c in df.columns), None)
    if cols["text"] is None:
        raise KeyError(f"No text column found (tried {CANDIDATES['text']}). Columns: {list(df.columns)}")
    return cols


_MULTIPLIERS = {"k": 1e3, "n": 1e3, "nghìn": 1e3, "ngàn": 1e3, "m": 1e6, "tr": 1e6, "triệu": 1e6, "b": 1e9, "tỷ": 1e9}


def parse_count(value) -> float:
    """'1,2K' -> 1200, '3.4k' -> 3400, '2 tr' -> 2e6, '1.234' -> 1234, '' / 'unknown' -> NaN."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value) if value >= 0 else np.nan
    s = str(value).strip().lower().replace(" ", " ")
    if s in ("", "nan", "none", "null", "unknown", "-"):
        return np.nan
    m = re.match(r"^([\d.,]+)\s*([a-zà-ỹ]*)", s)
    if not m:
        return np.nan
    number, unit = m.group(1).rstrip(".,"), m.group(2)
    if re.fullmatch(r"\d{1,3}(?:[.,]\d{3})+", number) and unit not in _MULTIPLIERS:
        num = float(re.sub(r"[.,]", "", number))           # thousands separators: '1.234' / '12,345,678'
    elif re.fullmatch(r"\d+(?:[.,]\d+)?", number):
        num = float(number.replace(",", "."))              # plain / decimal: '12', '12.0' (float CSV), '1,2'
    else:
        return np.nan
    if unit in _MULTIPLIERS:
        num *= _MULTIPLIERS[unit]
    elif unit:
        return np.nan
    return float(round(num))


def parse_timestamp(value) -> str:
    """Unix seconds / milliseconds or date string -> 'YYYY-MM-DD HH:MM:SS' (empty if invalid)."""
    if value is None or (isinstance(value, float) and np.isnan(value)) or str(value).strip() == "":
        return ""
    s = str(value).strip()
    try:
        if re.fullmatch(r"\d{9,13}(\.\d+)?", s):
            num = float(s)
            ts = pd.to_datetime(num, unit="ms" if num > 1e11 else "s")
        else:
            day_first = bool(re.match(r"^\d{1,2}[/.-]\d{1,2}[/.-]\d{4}", s))
            ts = pd.to_datetime(s, dayfirst=day_first)
        if ts.tzinfo is not None:
            ts = ts.tz_localize(None)
        return ts.strftime("%Y-%m-%d %H:%M:%S")
    except (ValueError, OverflowError, TypeError):
        return ""


def parse_image_field(value) -> List[str]:
    """'a.jpg' | 'a.jpg;b.jpg' | "['url1', 'url2']" | list -> list of strings."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, (list, tuple)):
        return [str(v).strip() for v in value if str(v).strip()]
    s = str(value).strip()
    if not s or s.lower() in ("nan", "none", "null", "[]", "0"):
        return []
    if s.startswith("["):
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, (list, tuple)):
                return [str(v).strip() for v in parsed if str(v).strip()]
        except (ValueError, SyntaxError):
            pass
    return [p.strip() for p in re.split(r"[;|\n]|,(?=\s*(?:https?://|[\w-]+\.(?:jpe?g|png|webp|gif)))", s) if p.strip()]


class ImageIndex:
    """Lookup of image files in a directory by file name, stem and post-id prefix."""

    def __init__(self, image_dir: Optional[str]) -> None:
        self.image_dir = image_dir
        self.by_name: Dict[str, str] = {}
        self.by_stem: Dict[str, List[str]] = {}
        if image_dir and os.path.isdir(image_dir):
            for root, _, files in os.walk(image_dir):
                for name in files:
                    if not name.lower().endswith(IMAGE_EXTENSIONS):
                        continue
                    rel = os.path.relpath(os.path.join(root, name), image_dir).replace("\\", "/")
                    self.by_name.setdefault(name.lower(), rel)
                    stem = os.path.splitext(name)[0].lower()
                    base = re.split(r"[_-]\d+$", stem)[0]  # '<id>_0' -> '<id>'
                    self.by_stem.setdefault(base, []).append(rel)
            for v in self.by_stem.values():
                v.sort()
            logger.info(f"Indexed {len(self.by_name)} image files in {image_dir}")
        elif image_dir:
            logger.warning(f"Image directory not found: {image_dir} (image paths will be kept as given)")

    def resolve(self, refs: List[str], post_id: str) -> List[str]:
        """Paths relative to image_dir (or raw references if no directory index is available)."""
        found: List[str] = []
        for ref in refs:
            name = os.path.basename(urlparse(ref).path) if re.match(r"^https?://", ref, re.I) else os.path.basename(ref)
            if self.by_name:
                hit = self.by_name.get(name.lower())
                if hit:
                    found.append(hit)
            elif not re.match(r"^https?://", ref, re.I):
                found.append(ref)  # no index: trust the given local path
        if not found and self.by_stem:
            found = list(self.by_stem.get(str(post_id).lower(), []))
        return list(dict.fromkeys(found))[:3]


# ----------------------------------------------------------------------
# Conversion
# ----------------------------------------------------------------------
def convert(
    df: pd.DataFrame,
    image_dir: Optional[str] = None,
    overrides: Optional[Dict[str, Optional[str]]] = None,
    id_prefix: str = "REINTEL_",
    keep_unlabeled: bool = False,
    source_file: str = "",
) -> pd.DataFrame:
    df = df.reset_index(drop=True)
    cols = resolve_columns(df, overrides or {})
    logger.info(f"Column mapping: {cols}")
    index = ImageIndex(image_dir)

    raw_ids = df[cols["id"]].astype(str) if cols["id"] else pd.Series([str(i) for i in range(len(df))])
    out = pd.DataFrame({c: [""] * len(df) for c in STANDARD_COLUMNS})
    out["sample_id"] = [f"{id_prefix}{i}" for i in raw_ids]
    out["platform"] = "Facebook"
    out["source"] = df[cols["user"]].fillna("").astype(str) if cols["user"] else ""
    out["timestamp"] = df[cols["timestamp"]].apply(parse_timestamp) if cols["timestamp"] else ""
    out["text"] = df[cols["text"]].fillna("").astype(str)
    out["language"] = "vi"

    for key, target in (("like", "like_count"), ("comment", "comment_count"), ("share", "share_count")):
        out[target] = df[cols[key]].apply(parse_count) if cols[key] else np.nan

    image_refs = df[cols["image"]].apply(parse_image_field) if cols["image"] else pd.Series([[]] * len(df))
    n_with_image = 0
    for i, (refs, pid) in enumerate(zip(image_refs, raw_ids)):
        urls = [r for r in refs if re.match(r"^https?://", r, re.I)]
        for slot, url in enumerate(urls[:3], start=1):
            out.at[i, f"image_{slot}_url"] = url
        paths = index.resolve(refs, pid)
        for slot, path in enumerate(paths, start=1):
            out.at[i, f"image_{slot}_local_path"] = path
        n_with_image += bool(paths)

    if cols["label"]:
        labels = pd.to_numeric(df[cols["label"]], errors="coerce")
        out["label"] = labels.map({1: 1, 0: 0})
        out["raw_label"] = labels.map({1: "unreliable", 0: "reliable"}).fillna("")
    else:
        out["label"] = np.nan
    out["label_source"] = "ReINTEL (VLSP 2020)"
    out["label_confidence"] = 1.0
    out["annotation_status"] = "reviewed"
    out["notes"] = f"converted from {os.path.basename(source_file)}" if source_file else "converted from ReINTEL"

    n_total = len(out)
    unlabeled = out["label"].isna()
    if unlabeled.any() and not keep_unlabeled:
        logger.info(f"Dropping {int(unlabeled.sum())} unlabeled rows (e.g. the public test file); "
                    f"use --keep_unlabeled to keep them for inference.")
        out = out[~unlabeled].reset_index(drop=True)

    empty_text = out["text"].str.strip() == ""
    report = {
        "rows_in": n_total,
        "rows_out": len(out),
        "label_counts": out["label"].value_counts(dropna=False).to_dict(),
        "with_image_file": n_with_image,
        "empty_text": int(empty_text.sum()),
        "missing_counts": {c: int(out[c].isna().sum()) for c in ("like_count", "comment_count", "share_count")},
        "missing_timestamp": int((out["timestamp"] == "").sum()),
        "unique_sources": int(out["source"].replace("", np.nan).nunique()),
    }
    logger.info(f"Conversion report: {report}")
    if n_with_image == 0 and cols["image"] is None and not index.by_stem:
        logger.warning("No image found for any post: check --image_dir / --image_col (text+social still usable).")
    out.attrs["report"] = report
    return out


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    parser = argparse.ArgumentParser(description="Convert ReINTEL (VLSP 2020) to the VMTF-Net schema")
    parser.add_argument("--input", nargs="+", required=True, help="One or more ReINTEL files (csv/json/jsonl/xlsx)")
    parser.add_argument("--output", required=True, help="Output CSV (55-column schema)")
    parser.add_argument("--image_dir", default=None, help="Directory with ReINTEL images (searched recursively)")
    parser.add_argument("--id_prefix", default="REINTEL_")
    parser.add_argument("--keep_unlabeled", action="store_true", help="Keep rows without label (inference only)")
    for key in CANDIDATES:
        parser.add_argument(f"--{key}_col", default=None, help=f"Override the {key} column name")
    args = parser.parse_args()

    overrides = {k: getattr(args, f"{k}_col") for k in CANDIDATES}
    parts = []
    for path in args.input:
        logger.info(f"Reading {path}")
        parts.append(convert(read_any(path), args.image_dir, overrides, args.id_prefix,
                             args.keep_unlabeled, source_file=path))
    result = pd.concat(parts, ignore_index=True)

    dup = result["sample_id"].duplicated()
    if dup.any():
        logger.warning(f"{int(dup.sum())} duplicated ids across input files: suffixing them to keep every row.")
        result.loc[dup, "sample_id"] = result.loc[dup, "sample_id"] + "_dup" + result.index[dup].astype(str)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    result.to_csv(args.output, index=False, encoding="utf-8")
    print(f"\nSaved {len(result)} rows to {args.output}")
    print(f"Label counts: {result['label'].value_counts(dropna=False).to_dict()}")
    print(f"Posts with a local image: {(result['image_1_local_path'] != '').sum()} / {len(result)}")
    print("Next: python scripts/validate_and_preprocess_dataset.py --input_csv "
          f"{args.output} --image_dir {args.image_dir or 'data/images'} --output_dir data")


if __name__ == "__main__":
    main()
