#!/usr/bin/env python3
"""
Dataset Validation, Preprocessing, and Splitting Script for VMTF-Net.
===================================================================
Features:
1. Validates standard 55-column schema compliance.
2. Checks image file presence on disk and computes missing SHA-256 and pHash (images 1-3).
3. Fills ONLY missing social ratios:
   - like_share_ratio = like_count / (share_count + 1)
   - comment_share_ratio = comment_count / (share_count + 1)
   - engagement_ratio = (like_count + comment_count + share_count * 2) / (like_count + 1)
4. Cleans template/example rows (ID starts with 'EXAMPLE_') and keeps only labels 0/1.
5. Anti-leakage grouping (union-find): rows sharing a duplicate_group / claim_id, an identical
   image (SHA-256), a (near-)identical image (pHash) or the same normalized text end up in the
   same group, so duplicates can never be split between train and test.
6. Splitting strategies:
   - group  : stratified (by label) group split 70/15/15 (default)
   - source : whole sources/pages are held out (tests generalization to unseen pages)
   - time   : chronological split (train on older posts, test on newer ones)
   - --kfold K : stratified group K-fold (fold i = test, fold i+1 = val, rest = train)
7. Prints a leakage report (group / source overlap between splits).
"""

import argparse
import hashlib
import logging
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("DatasetValidator")

# Expected 55 Standard Columns
STANDARD_COLUMNS = [
    "sample_id", "platform", "source", "source_url", "timestamp", "crawl_timestamp",
    "text", "language",
    "image_1_url", "image_1_local_path", "image_1_width", "image_1_height", "image_1_sha256", "image_1_phash",
    "image_2_url", "image_2_local_path", "image_2_width", "image_2_height", "image_2_sha256", "image_2_phash",
    "image_3_url", "image_3_local_path", "image_3_width", "image_3_height", "image_3_sha256", "image_3_phash",
    "like_count", "comment_count", "share_count", "angry_count", "haha_count",
    "like_share_ratio", "comment_share_ratio", "engagement_ratio",
    "rebuttal_comment_1", "rebuttal_comment_1_likes", "rebuttal_comment_1_timestamp",
    "rebuttal_comment_2", "rebuttal_comment_2_likes", "rebuttal_comment_2_timestamp",
    "rebuttal_comment_3", "rebuttal_comment_3_likes", "rebuttal_comment_3_timestamp",
    "label", "raw_label", "label_source", "label_confidence",
    "evidence_url", "evidence_note", "claim_id", "duplicate_group",
    "data_quality", "annotation_status", "annotator", "notes"
]

IMAGE_SLOTS = [1, 2, 3]


def compute_sha256(filepath: str) -> Optional[str]:
    """Compute SHA-256 hash of an image file."""
    if not os.path.exists(filepath):
        return None
    h = hashlib.sha256()
    try:
        with open(filepath, "rb") as f:
            while chunk := f.read(8192):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


def compute_image_meta(filepath: str) -> Tuple[Optional[str], Optional[int], Optional[int]]:
    """Return (pHash, width, height) if PIL / imagehash are available."""
    try:
        from PIL import Image
    except ImportError:
        return None, None, None
    try:
        with Image.open(filepath) as img:
            width, height = img.size
            try:
                import imagehash
                phash = str(imagehash.phash(img))
            except ImportError:
                phash = None
            return phash, width, height
    except Exception:
        return None, None, None


def parse_label(val) -> Optional[int]:
    """Normalize label values into binary 0 or 1 (anything else is rejected)."""
    if pd.isna(val) or val is None or str(val).strip() == "":
        return None
    val_str = str(val).strip().lower()
    if val_str in ["1", "1.0", "fake", "true_fake", "tin_gia", "unreliable"]:
        return 1
    if val_str in ["0", "0.0", "real", "legit", "tin_that", "reliable"]:
        return 0
    return None


def _is_empty(val) -> bool:
    return val is None or pd.isna(val) or str(val).strip() == ""


def normalize_text(text) -> str:
    """Lowercase, drop URLs / punctuation / extra spaces (for exact duplicate detection)."""
    if _is_empty(text):
        return ""
    t = str(text).lower()
    t = re.sub(r"https?://\S+|www\.\S+", " ", t)
    t = re.sub(r"[^\w\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def validate_and_preprocess(
    input_csv: str,
    image_dir: str = "data/images",
    filter_unreviewed: bool = False,
    auto_hash: bool = True,
) -> pd.DataFrame:
    """Validate, clean, enrich, and format the dataset."""
    logger.info(f"Loading raw dataset from: {input_csv}")
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input file not found: {input_csv}")

    df = pd.read_csv(input_csv)
    logger.info(f"Initial raw rows: {len(df)}")

    # 1. Check Missing Standard Columns
    missing_cols = [c for c in STANDARD_COLUMNS if c not in df.columns]
    if missing_cols:
        logger.warning(f"Adding missing standard columns with empty defaults: {missing_cols}")
        for c in missing_cols:
            df[c] = np.nan
    extra_cols = [c for c in df.columns if c not in STANDARD_COLUMNS]
    if extra_cols:
        logger.info(f"Dropping non-standard columns: {extra_cols}")
    df = df[STANDARD_COLUMNS].copy()

    # 2. Filter out template/mock example rows (starts with 'EXAMPLE_')
    example_mask = df["sample_id"].astype(str).str.startswith("EXAMPLE_")
    if example_mask.any():
        logger.info(f"Filtered out {example_mask.sum()} placeholder rows (ID starts with 'EXAMPLE_').")
        df = df[~example_mask].reset_index(drop=True)

    # Duplicate sample ids would silently corrupt prediction files
    dup_ids = df["sample_id"].astype(str).duplicated(keep="first")
    if dup_ids.any():
        logger.warning(f"Dropping {int(dup_ids.sum())} rows with duplicated sample_id.")
        df = df[~dup_ids].reset_index(drop=True)

    # 3. Filter unreviewed rows if requested
    if filter_unreviewed:
        valid_status = ["reviewed", "verified", "accepted"]
        status_mask = df["annotation_status"].astype(str).str.lower().isin(valid_status)
        logger.info(f"Filtering by annotation_status in {valid_status}: {status_mask.sum()} kept / {len(df)} total.")
        df = df[status_mask].reset_index(drop=True)

    # 4. Normalize Labels (strictly 0/1)
    logger.info("Normalizing labels...")
    df["label"] = df["label"].apply(parse_label)
    unlabeled_mask = df["label"].isna()
    if unlabeled_mask.any():
        logger.warning(f"Dropping {unlabeled_mask.sum()} rows with missing or invalid labels (only 0/1 accepted).")
        df = df[~unlabeled_mask].reset_index(drop=True)
    df["label"] = df["label"].astype(int)
    df["raw_label"] = df["raw_label"].fillna(df["label"].map({1: "fake", 0: "real"}))

    # 5. Clean social counts and fill ONLY missing derived ratios
    logger.info("Computing and verifying social engagement features...")
    for sc in ["like_count", "comment_count", "share_count", "angry_count", "haha_count"]:
        df[sc] = pd.to_numeric(df[sc], errors="coerce").fillna(0.0).clip(lower=0.0)

    likes, comments, shares = df["like_count"], df["comment_count"], df["share_count"]
    derived = {
        "like_share_ratio": likes / (shares + 1.0),
        "comment_share_ratio": comments / (shares + 1.0),
        "engagement_ratio": (likes + comments + shares * 2.0) / (likes + 1.0),
    }
    for name, values in derived.items():
        existing = pd.to_numeric(df[name], errors="coerce")
        n_fill = int(existing.isna().sum())
        df[name] = existing.fillna(values.round(4))
        if n_fill:
            logger.info(f"Filled {n_fill} missing values of '{name}'.")

    # 6. Verify Images on Disk & Auto-Hash (all image slots)
    logger.info(f"Verifying image files in directory: {image_dir}")
    found_images, missing_primary = 0, 0
    for idx, row in df.iterrows():
        for slot in IMAGE_SLOTS:
            img_name = row.get(f"image_{slot}_local_path")
            if _is_empty(img_name):
                if slot == 1:
                    missing_primary += 1
                continue
            img_path = str(img_name).strip()
            if not os.path.isabs(img_path) and not os.path.exists(img_path):
                img_path = os.path.join(image_dir, img_path)
            if not os.path.exists(img_path):
                if slot == 1:
                    missing_primary += 1
                continue
            found_images += 1
            if not auto_hash:
                continue
            if _is_empty(row.get(f"image_{slot}_sha256")):
                df.at[idx, f"image_{slot}_sha256"] = compute_sha256(img_path)
            if _is_empty(row.get(f"image_{slot}_phash")) or _is_empty(row.get(f"image_{slot}_width")):
                phash, width, height = compute_image_meta(img_path)
                if phash and _is_empty(row.get(f"image_{slot}_phash")):
                    df.at[idx, f"image_{slot}_phash"] = phash
                if width and _is_empty(row.get(f"image_{slot}_width")):
                    df.at[idx, f"image_{slot}_width"] = width
                    df.at[idx, f"image_{slot}_height"] = height

    logger.info(
        f"Image scan completed: {found_images} image files found; "
        f"{missing_primary}/{len(df)} posts have no usable primary image."
    )
    logger.info(f"Cleaned dataset ready: {len(df)} samples total.")
    return df


# ----------------------------------------------------------------------
# Anti-leakage grouping
# ----------------------------------------------------------------------
class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def _union_by_key(uf: _UnionFind, keys: List[Optional[str]]) -> int:
    """Union all rows sharing a non-empty key. Returns number of merges."""
    first: Dict[str, int] = {}
    merges = 0
    for i, k in enumerate(keys):
        if k is None:
            continue
        if k in first:
            if uf.find(first[k]) != uf.find(i):
                merges += 1
            uf.union(first[k], i)
        else:
            first[k] = i
    return merges


def _popcount64(x: np.ndarray) -> np.ndarray:
    if hasattr(np, "bitwise_count"):  # numpy >= 2.0
        return np.bitwise_count(x)
    return np.unpackbits(x.view(np.uint8).reshape(*x.shape, 8), axis=-1).sum(axis=-1)


def _union_near_duplicate_phash(uf: _UnionFind, phashes: List[Optional[str]], max_distance: int) -> int:
    """Union rows whose pHash Hamming distance <= max_distance (chunked O(n^2))."""
    idx = [i for i, p in enumerate(phashes) if p is not None and re.fullmatch(r"[0-9a-f]{16}", p)]
    if len(idx) < 2:
        return 0
    values = np.array([int(phashes[i], 16) for i in idx], dtype=np.uint64)
    merges = 0
    chunk = 512 if hasattr(np, "bitwise_count") else 64  # unpackbits fallback uses 64x more memory
    for start in range(0, len(values), chunk):
        block = values[start:start + chunk]
        dist = _popcount64(block[:, None] ^ values[None, :])
        rows, cols = np.nonzero(dist <= max_distance)
        for r, c in zip(rows, cols):
            a, b = idx[start + r], idx[c]
            if a < b and uf.find(a) != uf.find(b):
                uf.union(a, b)
                merges += 1
    return merges


def assign_leakage_groups(
    df: pd.DataFrame,
    text_dedup: bool = True,
    phash_max_distance: int = 0,
    min_text_len: int = 20,
) -> pd.DataFrame:
    """Compute `duplicate_group` so that duplicated content always shares a split."""
    n = len(df)
    uf = _UnionFind(n)
    stats = {}

    def _clean(series: pd.Series) -> List[Optional[str]]:
        return [None if _is_empty(v) else str(v).strip().lower() for v in series]

    stats["duplicate_group/claim_id"] = _union_by_key(uf, _clean(df["duplicate_group"]))
    stats["duplicate_group/claim_id"] += _union_by_key(uf, _clean(df["claim_id"]))

    # The same image may sit in any slot of another post
    sha_keys, phash_keys = {}, []
    for slot in IMAGE_SLOTS:
        for i, v in enumerate(_clean(df[f"image_{slot}_sha256"])):
            if v is not None:
                sha_keys.setdefault(v, []).append(i)
    merges = 0
    for rows in sha_keys.values():
        for r in rows[1:]:
            if uf.find(rows[0]) != uf.find(r):
                merges += 1
            uf.union(rows[0], r)
    stats["image_sha256"] = merges

    merges = 0
    for slot in IMAGE_SLOTS:
        phash_keys = _clean(df[f"image_{slot}_phash"])
        if phash_max_distance > 0:
            merges += _union_near_duplicate_phash(uf, phash_keys, phash_max_distance)
        else:
            merges += _union_by_key(uf, phash_keys)
    stats["image_phash"] = merges

    if text_dedup:
        norm = [normalize_text(t) for t in df["text"]]
        stats["normalized_text"] = _union_by_key(uf, [t if len(t) >= min_text_len else None for t in norm])

    roots = [uf.find(i) for i in range(n)]
    sample_ids = df["sample_id"].astype(str).tolist()
    df = df.copy()
    df["duplicate_group"] = [f"G_{sample_ids[r]}" for r in roots]
    logger.info(f"Leakage grouping merges: {stats} -> {df['duplicate_group'].nunique()} groups for {n} samples.")
    return df


# ----------------------------------------------------------------------
# Splitting
# ----------------------------------------------------------------------
def _group_table(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """One row per group with its size and majority label (used for stratification)."""
    g = df.groupby(group_col)["label"].agg(["size", "mean"]).reset_index()
    g["strat_label"] = (g["mean"] >= 0.5).astype(int)
    return g


def stratified_group_split(
    df: pd.DataFrame,
    group_col: str,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Group split where each label stratum is cut by sample count (no group crosses splits)."""
    rng = np.random.default_rng(seed)
    groups = _group_table(df, group_col)
    largest = groups["size"].max() / max(len(df), 1)
    if largest > min(val_ratio, 1.0 - train_ratio - val_ratio):
        logger.warning(
            f"The largest split unit holds {largest*100:.1f}% of the data (duplicates/sources glued together): "
            f"split proportions cannot be met exactly. Inspect `duplicate_group` / `source`."
        )
    assignment: Dict[str, str] = {}
    for _, stratum in groups.groupby("strat_label"):
        order = stratum.iloc[rng.permutation(len(stratum))]
        total = order["size"].sum()
        cum = order["size"].cumsum().values
        for g, c in zip(order[group_col].values, cum):
            frac = c / total
            if frac <= train_ratio + 1e-9:
                assignment[g] = "train"
            elif frac <= train_ratio + val_ratio + 1e-9:
                assignment[g] = "val"
            else:
                assignment[g] = "test"
    split = df[group_col].map(assignment)
    return (
        df[split == "train"].reset_index(drop=True),
        df[split == "val"].reset_index(drop=True),
        df[split == "test"].reset_index(drop=True),
    )


def temporal_split(
    df: pd.DataFrame,
    group_col: str,
    train_ratio: float,
    val_ratio: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chronological split on group start time (groups without timestamp go to train)."""
    ts = pd.to_datetime(df["timestamp"], errors="coerce")
    n_missing = int(ts.isna().sum())
    if n_missing:
        logger.warning(f"{n_missing} rows have no parseable timestamp: their groups are placed in train.")
    g_time = ts.groupby(df[group_col]).min()
    g_size = df.groupby(group_col).size()
    ordered = g_time.sort_values(na_position="first").index.tolist()
    total = len(df)
    cum = 0
    assignment = {}
    for g in ordered:
        cum += g_size[g]
        frac = cum / total
        if pd.isna(g_time[g]) or frac <= train_ratio + 1e-9:
            assignment[g] = "train"
        elif frac <= train_ratio + val_ratio + 1e-9:
            assignment[g] = "val"
        else:
            assignment[g] = "test"
    split = df[group_col].map(assignment)
    return (
        df[split == "train"].reset_index(drop=True),
        df[split == "val"].reset_index(drop=True),
        df[split == "test"].reset_index(drop=True),
    )


def stratified_group_kfold(df: pd.DataFrame, group_col: str, k: int, seed: int) -> np.ndarray:
    """Assign each row a fold id in [0, k) with balanced label strata and no group crossing folds."""
    rng = np.random.default_rng(seed)
    groups = _group_table(df, group_col)
    fold_of: Dict[str, int] = {}
    for _, stratum in groups.groupby("strat_label"):
        order = stratum.iloc[rng.permutation(len(stratum))]
        order = order.sort_values("size", ascending=False, kind="stable")
        load = np.zeros(k)
        for g, size in zip(order[group_col].values, order["size"].values):
            f = int(np.argmin(load))
            fold_of[g] = f
            load[f] += size
    return df[group_col].map(fold_of).values


def build_split_groups(df: pd.DataFrame, split_by: str) -> str:
    """Return the column defining the atomic split unit."""
    if split_by != "source":
        return "duplicate_group"
    # A source is held out as a whole; duplicates across sources glue those sources together
    source = df["source"].where(~df["source"].apply(_is_empty), df["sample_id"]).astype(str).str.strip().str.lower()
    uf = _UnionFind(len(df))
    _union_by_key(uf, source.tolist())
    _union_by_key(uf, df["duplicate_group"].astype(str).tolist())
    df["split_group"] = [f"S_{uf.find(i)}" for i in range(len(df))]
    return "split_group"


def leakage_report(splits: Dict[str, pd.DataFrame]) -> None:
    """Warn if any duplicate group / source appears in more than one split."""
    names = list(splits)
    for col in ("duplicate_group", "source"):
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = splits[names[i]], splits[names[j]]
                if col not in a or col not in b:
                    continue
                sa = set(a[col].dropna().astype(str))
                sb = set(b[col].dropna().astype(str))
                overlap = sa & sb
                if col == "duplicate_group" and overlap:
                    logger.error(f"LEAKAGE: {len(overlap)} duplicate groups shared by {names[i]} and {names[j]}!")
                elif col == "source" and overlap:
                    logger.info(
                        f"{len(overlap)} sources appear in both {names[i]} and {names[j]} "
                        f"(use --split_by source to evaluate on unseen pages)."
                    )


def print_summary(df: pd.DataFrame, splits: Dict[str, pd.DataFrame]) -> None:
    """Print structured summary report."""
    print("\n" + "=" * 60)
    print("       DATASET VALIDATION & SPLIT SUMMARY REPORT       ")
    print("=" * 60)
    print(f"Total Clean Samples:       {len(df)}")
    print(f"Total Unique Groups:       {df['duplicate_group'].nunique()}")
    real_count = (df["label"] == 0).sum()
    fake_count = (df["label"] == 1).sum()
    print(f"Overall Class Balance:     Real = {real_count} ({real_count/max(len(df),1)*100:.1f}%), "
          f"Fake = {fake_count} ({fake_count/max(len(df),1)*100:.1f}%)")
    print("-" * 60)
    for name, s in splits.items():
        fake = (s["label"] == 1).sum()
        print(f"{name:<12} {len(s):>6} samples (Real: {(s['label']==0).sum()}, Fake: {fake}, "
              f"Fake%: {fake/max(len(s),1)*100:.1f})")
    print("=" * 60 + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Validate, clean, enrich and split dataset for VMTF-Net")
    parser.add_argument("--input_csv", type=str, default="data/dataset_template.csv", help="Input raw CSV file")
    parser.add_argument("--output_clean_csv", type=str, default="data/dataset_clean.csv", help="Output path for cleaned full dataset")
    parser.add_argument("--image_dir", type=str, default="data/images", help="Directory where image files reside")
    parser.add_argument("--output_dir", type=str, default="data", help="Output directory for train/val/test splits")
    parser.add_argument("--filter_unreviewed", action="store_true", help="Only keep reviewed/verified samples")
    parser.add_argument("--no_split", action="store_true", help="Only clean and validate without generating splits")
    parser.add_argument("--no_hash", action="store_true", help="Skip SHA-256 / pHash computation")
    parser.add_argument("--split_by", type=str, default="group", choices=["group", "source", "time"],
                        help="group: stratified group split | source: unseen pages | time: chronological")
    parser.add_argument("--kfold", type=int, default=0, help="If > 1, also write stratified group K-fold splits")
    parser.add_argument("--train_ratio", type=float, default=0.70)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_text_dedup", action="store_true", help="Do not group identical normalized texts")
    parser.add_argument("--phash_max_distance", type=int, default=0,
                        help="Group images with pHash Hamming distance <= N (0 = exact match only; 4-6 = near duplicates)")
    return parser.parse_args()


def main():
    args = parse_args()
    assert abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) < 1e-5, "Split ratios must sum to 1.0"

    clean_df = validate_and_preprocess(
        input_csv=args.input_csv,
        image_dir=args.image_dir,
        filter_unreviewed=args.filter_unreviewed,
        auto_hash=not args.no_hash,
    )
    clean_df = assign_leakage_groups(
        clean_df,
        text_dedup=not args.no_text_dedup,
        phash_max_distance=args.phash_max_distance,
    )

    clean_df.to_csv(args.output_clean_csv, index=False, encoding="utf-8")
    logger.info(f"Saved cleaned dataset to: {args.output_clean_csv}")

    if args.no_split:
        return

    os.makedirs(args.output_dir, exist_ok=True)
    work_df = clean_df.copy()
    group_col = build_split_groups(work_df, args.split_by)

    if args.split_by == "time":
        train_df, val_df, test_df = temporal_split(work_df, group_col, args.train_ratio, args.val_ratio)
    else:
        train_df, val_df, test_df = stratified_group_split(
            work_df, group_col, args.train_ratio, args.val_ratio, args.seed
        )
    splits = {"train": train_df, "val": val_df, "test": test_df}
    for name, s in splits.items():
        if len(s) == 0:
            logger.error(f"Split '{name}' is EMPTY. Adjust ratios, --split_by or fix over-merged groups.")
        elif s["label"].nunique() < 2:
            logger.warning(f"Split '{name}' contains a single class: F1/AUC on it will be meaningless.")
        path = os.path.join(args.output_dir, f"{name}.csv")
        s.drop(columns=["split_group"], errors="ignore").to_csv(path, index=False, encoding="utf-8")
        logger.info(f"Saved {name} split ({len(s)}) to: {path}")
    leakage_report(splits)
    print_summary(clean_df, splits)

    if args.kfold and args.kfold > 1:
        if args.split_by == "time":
            logger.warning("K-fold ignores --split_by time (folds are not chronological).")
        folds = stratified_group_kfold(work_df, group_col, args.kfold, args.seed)
        for i in range(args.kfold):
            val_fold = (i + 1) % args.kfold
            fold_dir = os.path.join(args.output_dir, "folds", f"fold_{i}")
            os.makedirs(fold_dir, exist_ok=True)
            fold_splits = {
                "train": work_df[(folds != i) & (folds != val_fold)],
                "val": work_df[folds == val_fold],
                "test": work_df[folds == i],
            }
            for name, s in fold_splits.items():
                s.drop(columns=["split_group"], errors="ignore").to_csv(
                    os.path.join(fold_dir, f"{name}.csv"), index=False, encoding="utf-8"
                )
            logger.info(
                f"Fold {i}: train {len(fold_splits['train'])}, val {len(fold_splits['val'])}, "
                f"test {len(fold_splits['test'])} -> {fold_dir}"
            )


if __name__ == "__main__":
    main()
