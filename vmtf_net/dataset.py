"""
Dataset and DataLoader module for VMTF-Net (Vietnamese Multimodal Transformer Fusion Network).
Handles:
- Vietnamese word segmentation (VnCoreNLP / pyvi, PhoBERT only) and one-time pre-tokenization.
- Visual preprocessing for Vision Transformer (ViT) with an explicit `has_image` mask.
- Outlier-resistant normalization for social engagement metrics (serializable into checkpoints).
- Dynamic padding collate and reproducible multi-environment DataLoaders (Kaggle GPU, Colab, Local).
"""

import hashlib
import logging
import os
import random
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from PIL import Image, ImageFile
from sklearn.preprocessing import StandardScaler

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .config import VMTFConfig

# Permit loading of slightly damaged/truncated social media images
ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

BASE_COUNT_COLS: List[str] = ["like_count", "comment_count", "share_count", "angry_count", "haha_count"]


# ----------------------------------------------------------------------
# Labels
# ----------------------------------------------------------------------
def parse_label(val: Any) -> Optional[int]:
    """Strictly parse a label to 0 (real) or 1 (fake). Anything else returns None."""
    if val is None or pd.isna(val) or str(val).strip() == "":
        return None
    val_str = str(val).strip().lower()
    if val_str in ["1", "1.0", "fake", "true_fake", "tin_gia", "unreliable"]:
        return 1
    if val_str in ["0", "0.0", "real", "legit", "tin_that", "reliable"]:
        return 0
    return None


def compute_class_weights(labels: np.ndarray, num_classes: int = 2) -> torch.Tensor:
    """Inverse-frequency ('balanced') class weights: n_samples / (n_classes * count_c)."""
    counts = np.bincount(labels.astype(int), minlength=num_classes).astype(np.float64)
    counts[counts == 0] = 1.0
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


# ----------------------------------------------------------------------
# Word segmentation (PhoBERT expects word-segmented input, ViSoBERT does not)
# ----------------------------------------------------------------------
_SEGMENTER_CACHE: Dict[Tuple[str, Optional[str]], Callable[[str], str]] = {}


def _identity(text: str) -> str:
    return text


def build_word_segmenter(config: VMTFConfig) -> Tuple[Callable[[str], str], str]:
    """
    Resolve the word segmenter for the configured text backbone.
    PhoBERT was pre-trained on RDRSegmenter (VnCoreNLP) output, so VnCoreNLP is preferred;
    pyvi is a lighter fallback. Syllable-level models (ViSoBERT, XLM-R) get no segmentation.

    Returns:
        (segment_fn, resolved_mode)
    """
    mode = config.word_segmenter
    if mode == "auto":
        if "phobert" not in config.text_model_name.lower():
            return _identity, "none"
        mode = "vncorenlp" if config.vncorenlp_dir else "pyvi"
    if mode == "none":
        return _identity, "none"

    key = (mode, config.vncorenlp_dir)
    if key in _SEGMENTER_CACHE:
        return _SEGMENTER_CACHE[key], mode

    if mode == "vncorenlp":
        try:
            import py_vncorenlp

            save_dir = os.path.abspath(config.vncorenlp_dir or "./vncorenlp")
            cwd = os.getcwd()  # py_vncorenlp changes the working directory on init
            try:
                if not os.path.isdir(os.path.join(save_dir, "models")):
                    os.makedirs(save_dir, exist_ok=True)
                    logger.info(f"Downloading VnCoreNLP models to {save_dir} ...")
                    py_vncorenlp.download_model(save_dir=save_dir)
                rdrsegmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir=save_dir)
            finally:
                os.chdir(cwd)

            def _vncorenlp_segment(text: str) -> str:
                if not text.strip():
                    return text
                return " ".join(rdrsegmenter.word_segment(text))

            _SEGMENTER_CACHE[key] = _vncorenlp_segment
            logger.info("VnCoreNLP RDRSegmenter enabled (matches PhoBERT pre-training).")
            return _vncorenlp_segment, mode
        except Exception as e:
            logger.warning(f"VnCoreNLP unavailable ({e}). Falling back to pyvi.")
            mode = "pyvi"
            key = (mode, config.vncorenlp_dir)

    if mode == "pyvi":
        try:
            from pyvi import ViTokenizer

            _SEGMENTER_CACHE[key] = ViTokenizer.tokenize
            logger.info("pyvi word segmenter enabled for PhoBERT.")
            return ViTokenizer.tokenize, mode
        except ImportError:
            logger.warning("pyvi not installed: PhoBERT will receive unsegmented text (lower accuracy).")
            return _identity, "none"

    return _identity, "none"


def load_tokenizer(config: VMTFConfig) -> Any:
    """Load the HuggingFace tokenizer of the text backbone (hub id or local path)."""
    from transformers import AutoTokenizer

    logger.info(f"Loading text tokenizer: {config.text_model_name}")
    try:
        return AutoTokenizer.from_pretrained(config.text_model_name, use_fast=False)
    except Exception as e:  # newer transformers may not ship the slow class anymore
        logger.warning(f"Slow tokenizer unavailable ({e}); falling back to the default tokenizer.")
        return AutoTokenizer.from_pretrained(config.text_model_name)


def load_image_processor(config: VMTFConfig) -> Optional[Any]:
    """Load the HuggingFace image processor (only its mean/std are used)."""
    try:
        from transformers import AutoImageProcessor

        return AutoImageProcessor.from_pretrained(config.visual_model_name)
    except Exception as e:
        logger.warning(f"Could not load AutoImageProcessor ({e}). Using mean/std = 0.5.")
        return None


# ----------------------------------------------------------------------
# Social features
# ----------------------------------------------------------------------
class SocialFeatureNormalizer:
    """
    Normalizes social engagement metrics.
    Social counts (likes, shares, comments) follow extreme power-law distributions.
    We apply log1p transform followed by StandardScaler fitted strictly on the training set
    to prevent data leakage and numerical instability. The fitted statistics are stored in
    the checkpoint so that evaluation reuses exactly the training-set normalization.
    """

    def __init__(self, feature_cols: List[str]) -> None:
        self.feature_cols = list(feature_cols)
        self.scaler: Optional[StandardScaler] = None

    def _log_values(self, df: pd.DataFrame) -> np.ndarray:
        numeric_df = df[self.feature_cols].copy().apply(pd.to_numeric, errors="coerce").fillna(0.0)
        raw_vals = numeric_df.values.astype(np.float64)
        # Apply log1p on non-negative values to compress scale of viral outlier counts
        return np.log1p(np.clip(raw_vals, a_min=0.0, a_max=None))

    def fit(self, df: pd.DataFrame) -> "SocialFeatureNormalizer":
        """Fit StandardScaler on log1p-transformed social features."""
        self.scaler = StandardScaler()
        self.scaler.fit(self._log_values(df))
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform social features using fitted scaler."""
        log_vals = self._log_values(df)
        if self.scaler is not None:
            log_vals = self.scaler.transform(log_vals)
        return log_vals.astype(np.float32)

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """Fit on DataFrame and transform."""
        self.fit(df)
        return self.transform(df)

    def state_dict(self) -> Dict[str, Any]:
        """Serializable statistics (stored inside checkpoints)."""
        if self.scaler is None:
            return {"feature_cols": self.feature_cols, "mean": None, "scale": None}
        return {
            "feature_cols": self.feature_cols,
            "mean": self.scaler.mean_.tolist(),
            "scale": self.scaler.scale_.tolist(),
        }

    @classmethod
    def from_state_dict(cls, state: Dict[str, Any]) -> "SocialFeatureNormalizer":
        """Rebuild a fitted normalizer from `state_dict()` output."""
        normalizer = cls(state["feature_cols"])
        if state.get("mean") is not None:
            scaler = StandardScaler()
            scaler.mean_ = np.asarray(state["mean"], dtype=np.float64)
            scaler.scale_ = np.asarray(state["scale"], dtype=np.float64)
            scaler.var_ = scaler.scale_ ** 2
            scaler.n_features_in_ = len(normalizer.feature_cols)
            normalizer.scaler = scaler
        return normalizer


def add_derived_social_features(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """Ensure base counts are numeric and fill missing derived ratios from base counts."""
    for base_col in BASE_COUNT_COLS:
        if base_col in df.columns:
            df[base_col] = pd.to_numeric(df[base_col], errors="coerce").fillna(0.0).clip(lower=0.0)

    def _col(name: str) -> np.ndarray:
        return df[name].values if name in df.columns else np.zeros(len(df))

    derived = {
        "like_share_ratio": lambda: _col("like_count") / (_col("share_count") + 1.0),
        "comment_share_ratio": lambda: _col("comment_count") / (_col("share_count") + 1.0),
        "engagement_ratio": lambda: (_col("like_count") + _col("comment_count") + _col("share_count") * 2.0)
        / (_col("like_count") + 1.0),
    }
    for name, compute in derived.items():
        if name not in feature_cols:
            continue
        values = pd.Series(compute(), index=df.index)
        if name not in df.columns:
            df[name] = values
        else:
            df[name] = pd.to_numeric(df[name], errors="coerce").fillna(values)

    for col in feature_cols:
        if col not in df.columns:
            logger.warning(f"Social column '{col}' missing from data: filled with 0.")
            df[col] = 0.0
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    return df


# ----------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------
class VMTFDataset(Dataset):
    """
    Multimodal Dataset for Vietnamese Fake News Detection.
    Loads and preprocesses:
    1. Text -> word-segmented, pre-tokenized PhoBERT / ViSoBERT input_ids (variable length).
    2. Image -> ViT RGB pixel_values [3, image_size, image_size] + has_image flag.
    3. Social -> Normalized numerical engagement vector [num_features].
    4. Label -> Binary fake/real classification target (0 or 1).
    """

    def __init__(
        self,
        data: Union[str, pd.DataFrame],
        config: VMTFConfig,
        tokenizer: Optional[Any] = None,
        image_processor: Optional[Any] = None,
        social_normalizer: Optional[SocialFeatureNormalizer] = None,
        is_train: bool = True,
    ) -> None:
        """
        Args:
            data: Filepath to CSV or pre-loaded pandas DataFrame.
            config: VMTFConfig configuration instance.
            tokenizer: Pretrained HuggingFace tokenizer (PhoBERT).
            image_processor: HuggingFace image processor (provides normalization mean/std).
            social_normalizer: Normalizer fitted on the TRAINING split. Required for val/test.
            is_train: Flag for training mode (enables augmentation).
        """
        self.config = config
        self.is_train = is_train

        # 1. Load Data
        if isinstance(data, str):
            if not os.path.exists(data):
                raise FileNotFoundError(f"Data CSV not found at: {data}")
            self.df = pd.read_csv(data)
        elif isinstance(data, pd.DataFrame):
            self.df = data.copy().reset_index(drop=True)
        else:
            raise TypeError("Data must be either a file path string or pandas DataFrame.")

        # Ensure ID column exists (supports 'sample_id', 'post_id', 'id', 'claim_id')
        self.id_col = self._resolve_column_name(["sample_id", "post_id", "id", "claim_id"], required=False)
        if self.id_col is None:
            self.df["post_id"] = [f"sample_{i}" for i in range(len(self.df))]
            self.id_col = "post_id"

        # Filter out template/mock example rows (e.g. EXAMPLE_000001)
        example_mask = self.df[self.id_col].astype(str).str.startswith("EXAMPLE_")
        if example_mask.any():
            logger.info(f"Filtered out {example_mask.sum()} template/example rows (ID starts with 'EXAMPLE_').")
            self.df = self.df[~example_mask].reset_index(drop=True)

        self.text_col = self._resolve_column_name(["text", "content", "post_text", "post_message", "title"])
        self.image_col = self._resolve_column_name(
            ["image_1_local_path", "image_path", "image", "img_path", "image_name"], required=False
        )
        self.alt_image_cols = [
            c for c in ["image_2_local_path", "image_3_local_path"] if c in self.df.columns and c != self.image_col
        ]
        self.label_col = self._resolve_column_name(["label", "fake_label", "is_fake", "target"], required=False)

        # 2. Labels: drop rows whose label is not exactly 0/1 (otherwise CUDA asserts / ragged batches)
        self.labels: Optional[np.ndarray] = None
        if self.label_col is not None:
            parsed = self.df[self.label_col].apply(parse_label)
            invalid = parsed.isna()
            if invalid.all():
                logger.warning(f"Column '{self.label_col}' has no valid 0/1 label: treating data as unlabeled.")
                self.label_col = None
        if self.label_col is not None:
            if invalid.any():
                logger.warning(
                    f"Dropping {int(invalid.sum())} rows with missing/invalid labels (only 0/1 accepted)."
                )
                self.df = self.df[~invalid].reset_index(drop=True)
                parsed = parsed[~invalid].reset_index(drop=True)
            self.labels = parsed.astype(int).values

        # 3. Text: segment + tokenize ONCE (instead of every __getitem__ call of every epoch)
        self.input_ids: Optional[List[List[int]]] = None
        self.segmenter_mode = "none"
        if config.uses_text:
            self.tokenizer = tokenizer if tokenizer is not None else load_tokenizer(config)
            self.word_segmenter, self.segmenter_mode = build_word_segmenter(config)
            self.input_ids = self._pretokenize()

        # 4. Images: resolve paths once, build deterministic eval / augmented train transforms
        self.image_paths: List[Optional[str]] = [None] * len(self.df)
        if config.uses_image:
            if image_processor is None:
                image_processor = load_image_processor(config)
            mean = list(getattr(image_processor, "image_mean", None) or [0.5, 0.5, 0.5])
            std = list(getattr(image_processor, "image_std", None) or [0.5, 0.5, 0.5])
            size = config.image_size
            if self.is_train:
                # Same full-frame squash as eval first, then a mild crop: otherwise wide images get
                # center-cropped in training but squashed at test time (train/test mismatch).
                # No horizontal flip: many fake-news images contain text (screenshots, captions).
                self.image_transform = transforms.Compose([
                    transforms.Resize((size, size)),
                    transforms.RandomResizedCrop((size, size), scale=(0.85, 1.0), ratio=(0.9, 1.1)),
                    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean, std=std),
                ])
            else:
                self.image_transform = transforms.Compose([
                    transforms.Resize((size, size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean, std=std),
                ])
            self.image_paths = [self._find_image_path(row) for _, row in self.df.iterrows()]
            n_missing = sum(p is None for p in self.image_paths)
            if n_missing:
                logger.warning(
                    f"{n_missing}/{len(self.df)} samples have no image file on disk "
                    f"(handled with has_image=0, not with a fake placeholder signal)."
                )

        # 5. Social Features Normalization & Auto-completion
        self.df = add_derived_social_features(self.df, config.social_feature_cols)
        if social_normalizer is not None:
            self.social_normalizer = social_normalizer
            self.normalized_social = self.social_normalizer.transform(self.df)
        else:
            if not is_train:
                logger.warning(
                    "No social_normalizer supplied for a non-training split: fitting on this split. "
                    "Pass the training normalizer to avoid leakage."
                )
            self.social_normalizer = SocialFeatureNormalizer(config.social_feature_cols)
            self.normalized_social = self.social_normalizer.fit_transform(self.df)

        self.ids: List[str] = self.df[self.id_col].astype(str).tolist()
        self.has_image_flags: np.ndarray = np.array([p is not None for p in self.image_paths], dtype=bool)

    def _resolve_column_name(self, candidates: List[str], required: bool = True) -> Optional[str]:
        """Find the first matching column name from a list of candidates."""
        for name in candidates:
            if name in self.df.columns:
                return name
        if required:
            raise KeyError(f"None of expected candidate columns {candidates} were found in data.")
        return None

    def __len__(self) -> int:
        """Return total number of samples in the dataset."""
        return len(self.df)

    def _pretokenize(self) -> List[List[int]]:
        """Word-segment and tokenize all texts once, optionally caching to disk."""
        texts = [
            "" if pd.isna(t) else str(t)
            for t in self.df[self.text_col].tolist()
        ]

        cache_path = None
        if self.config.cache_dir:
            digest = hashlib.md5()
            digest.update(
                f"{self.config.text_model_name}|{self.config.max_seq_len}|{self.segmenter_mode}".encode("utf-8")
            )
            for t in texts:
                digest.update(t.encode("utf-8"))
                digest.update(b"\x00")
            os.makedirs(self.config.cache_dir, exist_ok=True)
            cache_path = os.path.join(self.config.cache_dir, f"tokens_{digest.hexdigest()}.pt")
            if os.path.exists(cache_path):
                logger.info(f"Loaded pre-tokenized text from cache: {cache_path}")
                return torch.load(cache_path)

        logger.info(f"Pre-tokenizing {len(texts)} texts (segmenter: {self.segmenter_mode})...")
        segmented = [self.word_segmenter(t) for t in texts]
        input_ids: List[List[int]] = []
        chunk = 512
        for start in range(0, len(segmented), chunk):
            enc = self.tokenizer(
                segmented[start:start + chunk],
                max_length=self.config.max_seq_len,
                truncation=True,
                padding=False,
            )
            input_ids.extend(enc["input_ids"])

        if cache_path is not None:
            torch.save(input_ids, cache_path)
            logger.info(f"Saved pre-tokenized text cache: {cache_path}")
        return input_ids

    def _find_image_path(self, row: pd.Series) -> Optional[str]:
        """
        Return the first existing image path from primary and fallback columns, or None.
        Supports full path, path relative to config.image_dir, and bare filename.
        """
        candidate_cols = ([self.image_col] if self.image_col else []) + self.alt_image_cols
        for col in candidate_cols:
            val = row.get(col, None)
            if val is None or pd.isna(val) or str(val).strip() == "":
                continue
            c_path = str(val).strip()
            for p in (
                c_path,
                os.path.join(self.config.image_dir, c_path),
                os.path.join(self.config.image_dir, os.path.basename(c_path)),
            ):
                if os.path.isfile(p):
                    return p
        return None

    def _load_image(self, path: Optional[str]) -> Tuple[Optional[Image.Image], bool]:
        """Load an RGB image. Returns (image, ok); ok=False when missing or corrupted."""
        if path is None:
            return None, False
        try:
            with Image.open(path) as img:
                return img.convert("RGB"), True
        except Exception as e:
            logger.warning(f"Error loading image {path}: {e}. Treated as missing.")
            return None, False

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Retrieve a single preprocessed multimodal sample.

        Returns:
            Dict containing:
                - 'id': str
                - 'input_ids': torch.LongTensor [seq_len] (unpadded; padded in collate)
                - 'pixel_values': torch.FloatTensor [3, image_size, image_size]
                - 'has_image': torch.FloatTensor scalar (1 = real image, 0 = missing)
                - 'social_features': torch.FloatTensor [num_social_features]
                - 'label': torch.LongTensor scalar (if labels available)
        """
        item: Dict[str, Any] = {
            "id": self.ids[idx],
            "social_features": torch.from_numpy(self.normalized_social[idx]),
        }

        if self.input_ids is not None:
            item["input_ids"] = torch.tensor(self.input_ids[idx], dtype=torch.long)

        if self.config.uses_image:
            image, ok = self._load_image(self.image_paths[idx])
            if ok:
                item["pixel_values"] = self.image_transform(image)
            else:
                # Zero tensor; the model ignores it through has_image = 0
                size = self.config.image_size
                item["pixel_values"] = torch.zeros(3, size, size)
            item["has_image"] = torch.tensor(1.0 if ok else 0.0)

        if self.labels is not None:
            item["label"] = torch.tensor(int(self.labels[idx]), dtype=torch.long)

        return item


class VMTFCollate:
    """
    Collate function batching multimodal samples with dynamic padding
    (pads text to the longest sequence in the batch instead of max_seq_len).
    """

    def __init__(self, pad_token_id: int = 1) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Returns:
            Dict containing:
                - 'id': List[str] of length B
                - 'input_ids': torch.LongTensor [B, L_batch]
                - 'attention_mask': torch.LongTensor [B, L_batch]
                - 'pixel_values': torch.FloatTensor [B, 3, image_size, image_size]
                - 'has_image': torch.FloatTensor [B]
                - 'social_features': torch.FloatTensor [B, num_features]
                - 'label': torch.LongTensor [B] (only if every sample has a label)
        """
        collated: Dict[str, Any] = {
            "id": [item["id"] for item in batch],
            "social_features": torch.stack([item["social_features"] for item in batch], dim=0),
        }

        if "input_ids" in batch[0]:
            max_len = max(item["input_ids"].size(0) for item in batch)
            input_ids = torch.full((len(batch), max_len), self.pad_token_id, dtype=torch.long)
            attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
            for i, item in enumerate(batch):
                n = item["input_ids"].size(0)
                input_ids[i, :n] = item["input_ids"]
                attention_mask[i, :n] = 1
            collated["input_ids"] = input_ids
            collated["attention_mask"] = attention_mask

        if "pixel_values" in batch[0]:
            collated["pixel_values"] = torch.stack([item["pixel_values"] for item in batch], dim=0)
            collated["has_image"] = torch.stack([item["has_image"] for item in batch], dim=0)

        if all("label" in item for item in batch):
            collated["label"] = torch.stack([item["label"] for item in batch], dim=0)

        return collated


def _seed_worker(worker_id: int) -> None:
    """Make numpy / random inside DataLoader workers deterministic."""
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _make_loader(dataset: VMTFDataset, config: VMTFConfig, batch_size: int, shuffle: bool, pad_id: int) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(config.seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        collate_fn=VMTFCollate(pad_token_id=pad_id),
        drop_last=shuffle and len(dataset) > batch_size,
        persistent_workers=config.num_workers > 0,
        worker_init_fn=_seed_worker,
        generator=generator,
    )


def _pad_id(tokenizer: Optional[Any]) -> int:
    pad = getattr(tokenizer, "pad_token_id", None) if tokenizer is not None else None
    return int(pad) if pad is not None else 1


def create_eval_loader(
    config: VMTFConfig,
    data: Union[str, pd.DataFrame],
    social_normalizer: SocialFeatureNormalizer,
    tokenizer: Optional[Any] = None,
    image_processor: Optional[Any] = None,
) -> DataLoader:
    """Build a deterministic loader for val/test data using the TRAINING social normalizer."""
    if config.uses_text and tokenizer is None:
        tokenizer = load_tokenizer(config)
    dataset = VMTFDataset(
        data=data,
        config=config,
        tokenizer=tokenizer,
        image_processor=image_processor,
        social_normalizer=social_normalizer,
        is_train=False,
    )
    return _make_loader(dataset, config, config.val_batch_size, shuffle=False, pad_id=_pad_id(tokenizer))


def create_dataloaders(
    config: VMTFConfig,
    train_data: Union[str, pd.DataFrame],
    val_data: Optional[Union[str, pd.DataFrame]] = None,
    test_data: Optional[Union[str, pd.DataFrame]] = None,
    tokenizer: Optional[Any] = None,
    image_processor: Optional[Any] = None,
) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader], SocialFeatureNormalizer]:
    """
    Factory function to construct PyTorch DataLoaders for VMTF-Net.
    Ensures that social feature normalizers are fitted exclusively on train_data
    to avoid data leakage across splits.

    Returns:
        Tuple of (train_loader, val_loader, test_loader, fitted_social_normalizer)
    """
    # 1. Load HuggingFace Tokenizer and Image Processor once
    if config.uses_text and tokenizer is None:
        tokenizer = load_tokenizer(config)
    if config.uses_image and image_processor is None:
        image_processor = load_image_processor(config)
    pad_id = _pad_id(tokenizer)

    # 2. Build Training Dataset & Fit Social Normalizer
    train_dataset = VMTFDataset(
        data=train_data,
        config=config,
        tokenizer=tokenizer,
        image_processor=image_processor,
        social_normalizer=None,  # Will fit on train_data
        is_train=True,
    )
    social_normalizer = train_dataset.social_normalizer
    train_loader = _make_loader(train_dataset, config, config.batch_size, shuffle=True, pad_id=pad_id)

    # 3. Validation / Test DataLoaders reuse the fitted normalizer
    val_loader = None
    if val_data is not None:
        val_loader = create_eval_loader(config, val_data, social_normalizer, tokenizer, image_processor)

    test_loader = None
    if test_data is not None:
        test_loader = create_eval_loader(config, test_data, social_normalizer, tokenizer, image_processor)

    logger.info(
        f"DataLoaders successfully built: Train samples={len(train_dataset)}, "
        f"Val samples={len(val_loader.dataset) if val_loader is not None else 0}, "
        f"Test samples={len(test_loader.dataset) if test_loader is not None else 0}"
    )

    return train_loader, val_loader, test_loader, social_normalizer


# ----------------------------------------------------------------------
# Backward-Compatible & Master-Prompt Aliases
# ----------------------------------------------------------------------
VietnameseFakeNewsDataset = VMTFDataset
get_dataloaders = create_dataloaders
