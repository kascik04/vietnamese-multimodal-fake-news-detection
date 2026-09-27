#!/usr/bin/env python3
"""
Dataset Validation, Preprocessing, and Splitting Script for VMTF-Net.
===================================================================
Features:
1. Validates standard 55-column schema compliance.
2. Checks image file presence on disk and computes missing SHA-256 and pHash.
3. Automatically derives missing social ratios:
   - like_share_ratio = like_count / (share_count + 1)
   - comment_share_ratio = comment_count / (share_count + 1)
   - engagement_ratio = (like_count + comment_count + share_count * 2) / (like_count + 1)
4. Cleans template/example rows (ID starts with 'EXAMPLE_') and normalizes labels (0/1).
5. Splits dataset into Train (70%), Val (15%), Test (15%) with Group-Stratified
   logic based on `duplicate_group` / `claim_id` to prevent data leakage.
"""

import argparse
import hashlib
import logging
import os
import sys
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

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


def compute_phash(filepath: str) -> Optional[str]:
    """Compute perceptual hash if imagehash / PIL is available."""
    try:
        from PIL import Image
        import imagehash
        with Image.open(filepath) as img:
            return str(imagehash.phash(img))
    except ImportError:
        return None
    except Exception:
        return None


def parse_label(val) -> Optional[int]:
    """Normalize label values into binary 0 or 1."""
    if pd.isna(val) or val is None or str(val).strip() == "":
        return None
    val_str = str(val).strip().lower()
    if val_str in ["1", "1.0", "fake", "true_fake", "tin_gia"]:
        return 1
    if val_str in ["0", "0.0", "real", "legit", "tin_that"]:
        return 0
    try:
        f_val = float(val)
        if not np.isnan(f_val):
            return int(f_val)
    except (ValueError, TypeError):
        pass
    return None


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

    # Ensure column order aligns with standard
    df = df[STANDARD_COLUMNS].copy()

    # 2. Filter out template/mock example rows (starts with 'EXAMPLE_')
    if "sample_id" in df.columns:
        example_mask = df["sample_id"].astype(str).str.startswith("EXAMPLE_")
        if example_mask.any():
            logger.info(f"Filtered out {example_mask.sum()} placeholder rows (ID starts with 'EXAMPLE_').")
            df = df[~example_mask].reset_index(drop=True)

    # 3. Filter unreviewed rows if requested
    if filter_unreviewed and "annotation_status" in df.columns:
        valid_status = ["reviewed", "verified", "accepted"]
        status_mask = df["annotation_status"].astype(str).str.lower().isin(valid_status)
        logger.info(f"Filtering by annotation_status in {valid_status}: {status_mask.sum()} kept / {len(df)} total.")
        df = df[status_mask].reset_index(drop=True)

    # 4. Normalize Labels
    logger.info("Normalizing labels...")
    df["label"] = df["label"].apply(parse_label)
    unlabeled_mask = df["label"].isna()
    if unlabeled_mask.any():
        logger.warning(f"Dropping {unlabeled_mask.sum()} rows with missing or unparseable labels.")
        df = df[~unlabeled_mask].reset_index(drop=True)
    df["label"] = df["label"].astype(int)

    # Populate raw_label if missing
    df["raw_label"] = df["raw_label"].fillna(df["label"].map({1: "fake", 0: "real"}))

    # 5. Clean and compute Social Interaction Metrics & Ratios
    logger.info("Computing and verifying social engagement features...")
    social_counts = ["like_count", "comment_count", "share_count", "angry_count", "haha_count"]
    for sc in social_counts:
        df[sc] = pd.to_numeric(df[sc], errors="coerce").fillna(0.0).clip(lower=0.0)

    # Compute derived ratios if missing or null
    likes = df["like_count"].values
    comments = df["comment_count"].values
    shares = df["share_count"].values

    if df["like_share_ratio"].isna().all() or df["like_share_ratio"].isna().sum() > 0:
        df["like_share_ratio"] = np.round(likes / (shares + 1.0), 4)

    if df["comment_share_ratio"].isna().all() or df["comment_share_ratio"].isna().sum() > 0:
        df["comment_share_ratio"] = np.round(comments / (shares + 1.0), 4)

    if df["engagement_ratio"].isna().all() or df["engagement_ratio"].isna().sum() > 0:
        df["engagement_ratio"] = np.round((likes + comments + shares * 2.0) / (likes + 1.0), 4)

    # 6. Verify Images on Disk & Auto-Hash
    logger.info(f"Verifying image files in directory: {image_dir}")
    os.makedirs(image_dir, exist_ok=True)
    found_images = 0
    missing_images = 0

    for idx, row in df.iterrows():
        img_name = row.get("image_1_local_path")
        if pd.isna(img_name) or str(img_name).strip() == "":
            missing_images += 1
            continue

        img_path = str(img_name).strip()
        if not os.path.isabs(img_path):
            img_path = os.path.join(image_dir, img_path)

        if os.path.exists(img_path):
            found_images += 1
            if auto_hash:
                if pd.isna(row.get("image_1_sha256")) or str(row.get("image_1_sha256")).strip() == "":
                    df.at[idx, "image_1_sha256"] = compute_sha256(img_path)
                if pd.isna(row.get("image_1_phash")) or str(row.get("image_1_phash")).strip() == "":
                    ph = compute_phash(img_path)
                    if ph:
                        df.at[idx, "image_1_phash"] = ph
        else:
            missing_images += 1

    logger.info(f"Image scan completed: {found_images} found on disk, {missing_images} missing or empty.")

    # 7. Ensure Group ID exists for Grouped Splitting
    if "duplicate_group" not in df.columns or df["duplicate_group"].isna().all():
        if "claim_id" in df.columns and not df["claim_id"].isna().all():
            df["duplicate_group"] = df["claim_id"].fillna(df["sample_id"])
        else:
            df["duplicate_group"] = df["sample_id"]
    else:
        df["duplicate_group"] = df["duplicate_group"].fillna(df["sample_id"])

    logger.info(f"Cleaned dataset ready: {len(df)} samples total.")
    return df


def split_dataset(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split dataset into Train, Val, and Test sets.
    Uses group-based splitting by `duplicate_group` to prevent data leakage.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "Split ratios must sum to 1.0"
    np.random.seed(seed)

    # Unique groups
    unique_groups = df["duplicate_group"].unique()
    np.random.shuffle(unique_groups)

    n_groups = len(unique_groups)
    n_train = int(n_groups * train_ratio)
    n_val = int(n_groups * val_ratio)

    train_groups = set(unique_groups[:n_train])
    val_groups = set(unique_groups[n_train:n_train + n_val])
    test_groups = set(unique_groups[n_train + n_val:])

    train_df = df[df["duplicate_group"].isin(train_groups)].reset_index(drop=True)
    val_df = df[df["duplicate_group"].isin(val_groups)].reset_index(drop=True)
    test_df = df[df["duplicate_group"].isin(test_groups)].reset_index(drop=True)

    return train_df, val_df, test_df


def print_summary(df: pd.DataFrame, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """Print structured summary report."""
    print("\n" + "=" * 60)
    print("       DATASET VALIDATION & SPLIT SUMMARY REPORT       ")
    print("=" * 60)
    print(f"Total Clean Samples:       {len(df)}")
    print(f"Total Unique Groups:       {df['duplicate_group'].nunique()}")
    real_count = (df['label'] == 0).sum()
    fake_count = (df['label'] == 1).sum()
    print(f"Overall Class Balance:     Real = {real_count} ({real_count/len(df)*100:.1f}%), Fake = {fake_count} ({fake_count/len(df)*100:.1f}%)")
    print("-" * 60)
    print(f"Train Split:               {len(train_df)} samples (Real: {(train_df['label']==0).sum()}, Fake: {(train_df['label']==1).sum()})")
    print(f"Val Split:                 {len(val_df)} samples (Real: {(val_df['label']==0).sum()}, Fake: {(val_df['label']==1).sum()})")
    print(f"Test Split:                {len(test_df)} samples (Real: {(test_df['label']==0).sum()}, Fake: {(test_df['label']==1).sum()})")
    print("=" * 60 + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Validate, clean, enrich and split dataset for VMTF-Net")
    parser.add_argument("--input_csv", type=str, default="data/dataset_template.csv", help="Input raw CSV file")
    parser.add_argument("--output_clean_csv", type=str, default="data/dataset_clean.csv", help="Output path for cleaned full dataset")
    parser.add_argument("--image_dir", type=str, default="data/images", help="Directory where image files reside")
    parser.add_argument("--output_dir", type=str, default="data", help="Output directory for train/val/test splits")
    parser.add_argument("--filter_unreviewed", action="store_true", help="Only keep reviewed/verified samples")
    parser.add_argument("--no_split", action="store_true", help="Only clean and validate without generating splits")
    return parser.parse_args()


def main():
    args = parse_args()
    clean_df = validate_and_preprocess(
        input_csv=args.input_csv,
        image_dir=args.image_dir,
        filter_unreviewed=args.filter_unreviewed,
    )

    # Save cleaned full dataset
    clean_df.to_csv(args.output_clean_csv, index=False, encoding="utf-8")
    logger.info(f"Saved cleaned dataset to: {args.output_clean_csv}")

    if not args.no_split:
        train_df, val_df, test_df = split_dataset(clean_df)
        train_path = os.path.join(args.output_dir, "train.csv")
        val_path = os.path.join(args.output_dir, "val.csv")
        test_path = os.path.join(args.output_dir, "test.csv")

        train_df.to_csv(train_path, index=False, encoding="utf-8")
        val_df.to_csv(val_path, index=False, encoding="utf-8")
        test_df.to_csv(test_path, index=False, encoding="utf-8")

        logger.info(f"Saved Train Split ({len(train_df)}) to: {train_path}")
        logger.info(f"Saved Val Split   ({len(val_df)}) to: {val_path}")
        logger.info(f"Saved Test Split  ({len(test_df)}) to: {test_path}")

        print_summary(clean_df, train_df, val_df, test_df)


if __name__ == "__main__":
    main()
