"""
Dataset and DataLoader module for VMTF-Net (Vietnamese Multimodal Transformer Fusion Network).
Handles:
- Vietnamese text tokenization with PhoBERT / ViSoBERT AutoTokenizer.
- Visual preprocessing for Vision Transformer (ViT) with AutoImageProcessor / Torchvision fallback.
- Outlier-resistant normalization for social engagement metrics.
- Multi-environment PyTorch DataLoader creation (Kaggle GPU, Colab, Local).
"""

import logging
import os
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


class SocialFeatureNormalizer:
    """
    Normalizes social engagement metrics.
    Social counts (likes, shares, comments) follow extreme power-law distributions.
    We apply log1p transform followed by StandardScaler fitted strictly on the training set
    to prevent data leakage and numerical instability.
    """

    def __init__(self, feature_cols: List[str]) -> None:
        self.feature_cols = feature_cols
        self.scaler: Optional[StandardScaler] = None

    def fit(self, df: pd.DataFrame) -> "SocialFeatureNormalizer":
        """Fit StandardScaler on log1p-transformed social features."""
        raw_vals = df[self.feature_cols].copy().fillna(0.0).values.astype(np.float32)
        # Apply log1p on non-negative values to compress scale of viral outlier counts
        log_vals = np.log1p(np.clip(raw_vals, a_min=0.0, a_max=None))
        self.scaler = StandardScaler()
        self.scaler.fit(log_vals)
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform social features using fitted scaler."""
        raw_vals = df[self.feature_cols].copy().fillna(0.0).values.astype(np.float32)
        log_vals = np.log1p(np.clip(raw_vals, a_min=0.0, a_max=None))
        if self.scaler is not None:
            norm_vals = self.scaler.transform(log_vals)
        else:
            norm_vals = log_vals
        return norm_vals.astype(np.float32)

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """Fit on DataFrame and transform."""
        self.fit(df)
        return self.transform(df)


class VMTFDataset(Dataset):
    """
    Multimodal Dataset for Vietnamese Fake News Detection.
    Loads and preprocesses:
    1. Text -> PhoBERT / ViSoBERT input_ids & attention_mask [seq_len].
    2. Image -> ViT RGB pixel_values [3, image_size, image_size].
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
            image_processor: HuggingFace ViTImageProcessor / AutoImageProcessor.
            social_normalizer: Fitted SocialFeatureNormalizer instance.
            is_train: Flag for training mode (enables augmentation if specified).
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

        # Ensure ID column exists (supports 'post_id', 'id', 'sample_id')
        self.id_col = self._resolve_column_name(["post_id", "id", "sample_id"], required=False)
        if self.id_col is None:
            self.df["post_id"] = [f"sample_{i}" for i in range(len(self.df))]
            self.id_col = "post_id"

        # Ensure Text column exists (supports 'text', 'content', 'post_text', 'title')
        self.text_col = self._resolve_column_name(["text", "content", "post_text", "title"])
        
        # Ensure Image Path column exists (supports 'image_path', 'image', 'img_path', 'image_name')
        self.image_col = self._resolve_column_name(["image_path", "image", "img_path", "image_name"], required=False)

        # Ensure Label column exists if present (supports 'label', 'fake_label', 'is_fake', 'target')
        self.label_col = self._resolve_column_name(["label", "fake_label", "is_fake", "target"], required=False)


        # 2. Text Tokenizer & Segmenter Setup
        if tokenizer is None:
            from transformers import AutoTokenizer
            logger.info(f"Loading text tokenizer: {self.config.text_model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.text_model_name, use_fast=False)
        else:
            self.tokenizer = tokenizer

        # Optional pyvi Vietnamese word segmenter for PhoBERT (joins compound words e.g. "Bộ_Y_tế")
        self.use_word_segmentation = False
        try:
            from pyvi import ViTokenizer
            self.word_segmenter = ViTokenizer.tokenize
            self.use_word_segmentation = True
            logger.info("pyvi Vietnamese word segmenter enabled for PhoBERT optimization.")
        except ImportError:
            self.word_segmenter = lambda x: x
            logger.info("pyvi not installed. Using standard tokenization (install pyvi for best PhoBERT performance).")

        # 3. Image Preprocessor Setup
        self.image_processor = image_processor
        if self.image_processor is None:
            try:
                from transformers import AutoImageProcessor
                self.image_processor = AutoImageProcessor.from_pretrained(self.config.visual_model_name)
            except Exception as e:
                logger.warning(f"Could not load AutoImageProcessor ({e}). Using torchvision transforms fallback.")
                self.image_processor = None

        # Fallback / Augmentation transforms for ViT (224x224, standard normalization)
        if self.is_train:
            self.torch_transform = transforms.Compose([
                transforms.Resize((self.config.image_size, self.config.image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
        else:
            self.torch_transform = transforms.Compose([
                transforms.Resize((self.config.image_size, self.config.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])

        # 4. Social Features Normalization
        # Check available social feature columns in DataFrame
        valid_social_cols = [c for c in self.config.social_feature_cols if c in self.df.columns]
        if len(valid_social_cols) < len(self.config.social_feature_cols):
            missing_cols = set(self.config.social_feature_cols) - set(valid_social_cols)
            logger.info(f"Creating missing social feature columns with 0.0: {missing_cols}")
            for col in missing_cols:
                self.df[col] = 0.0

        if social_normalizer is not None:
            self.social_normalizer = social_normalizer
            self.normalized_social = self.social_normalizer.transform(self.df)
        else:
            # Fit on current dataset if no external normalizer supplied
            self.social_normalizer = SocialFeatureNormalizer(self.config.social_feature_cols)
            self.normalized_social = self.social_normalizer.fit_transform(self.df)

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

    def _load_image(self, img_path: Optional[str]) -> Image.Image:
        """
        Safely load image from relative or absolute file path.
        If file is missing or corrupted, returns a neutral placeholder image.
        """
        if not img_path or pd.isna(img_path) or str(img_path).strip() == "":
            return Image.new("RGB", (self.config.image_size, self.config.image_size), color=(128, 128, 128))

        # Check full path or relative to config.image_dir
        candidate_path = str(img_path).strip()
        if not os.path.isabs(candidate_path) and not os.path.exists(candidate_path):
            candidate_path = os.path.join(self.config.image_dir, candidate_path)

        if not os.path.exists(candidate_path):
            # Fallback placeholder image
            return Image.new("RGB", (self.config.image_size, self.config.image_size), color=(128, 128, 128))

        try:
            with Image.open(candidate_path) as img:
                return img.convert("RGB")
        except Exception as e:
            logger.warning(f"Error loading image {candidate_path}: {e}. Using neutral placeholder.")
            return Image.new("RGB", (self.config.image_size, self.config.image_size), color=(128, 128, 128))

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Retrieve a single preprocessed multimodal sample.
        
        Returns:
            Dict containing:
                - 'id': str
                - 'input_ids': torch.LongTensor [max_seq_len]
                - 'attention_mask': torch.LongTensor [max_seq_len]
                - 'pixel_values': torch.FloatTensor [3, image_size, image_size]
                - 'social_features': torch.FloatTensor [num_social_features]
                - 'label': torch.LongTensor scalar (optional if available)
        """
        row = self.df.iloc[idx]
        sample_id = str(row[self.id_col])

        # ------------------------------------------------------------------
        # 1. Text Modality Processing
        # ------------------------------------------------------------------
        raw_text = str(row[self.text_col]) if self.text_col and not pd.isna(row[self.text_col]) else ""
        text = self.word_segmenter(raw_text)
        
        text_encoding = self.tokenizer(
            text,
            max_length=self.config.max_seq_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = text_encoding["input_ids"].squeeze(0)          # [max_seq_len]
        attention_mask = text_encoding["attention_mask"].squeeze(0)  # [max_seq_len]

        # ------------------------------------------------------------------
        # 2. Visual Modality Processing
        # ------------------------------------------------------------------
        img_val = row[self.image_col] if self.image_col else None
        image = self._load_image(img_val)

        if self.image_processor is not None and not self.is_train:
            # Use HuggingFace processor if available
            processed_img = self.image_processor(images=image, return_tensors="pt")
            pixel_values = processed_img["pixel_values"].squeeze(0)  # [3, image_size, image_size]
        else:
            # Use torchvision transforms (supports data augmentation during training)
            pixel_values = self.torch_transform(image)               # [3, image_size, image_size]

        # ------------------------------------------------------------------
        # 3. Social Metrics Modality Processing
        # ------------------------------------------------------------------
        social_vec = torch.tensor(self.normalized_social[idx], dtype=torch.float32)  # [num_social_features]

        # ------------------------------------------------------------------
        # 4. Construct Output Item
        # ------------------------------------------------------------------
        item: Dict[str, Any] = {
            "id": sample_id,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "social_features": social_vec,
        }

        # Include label if available in dataset
        if self.label_col and not pd.isna(row[self.label_col]):
            item["label"] = torch.tensor(int(row[self.label_col]), dtype=torch.long)

        return item


class VMTFCollate:
    """
    Collate function to batch multiple multimodal samples into stacked PyTorch tensors.
    """

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Collates list of sample dicts into a batched dictionary.
        
        Args:
            batch: List of dictionaries from VMTFDataset.__getitem__
            
        Returns:
            Dict containing:
                - 'id': List[str] of length B
                - 'input_ids': torch.LongTensor [B, max_seq_len]
                - 'attention_mask': torch.LongTensor [B, max_seq_len]
                - 'pixel_values': torch.FloatTensor [B, 3, image_size, image_size]
                - 'social_features': torch.FloatTensor [B, num_features]
                - 'label': torch.LongTensor [B] (if present)
        """
        collated: Dict[str, Any] = {
            "id": [item["id"] for item in batch],
            "input_ids": torch.stack([item["input_ids"] for item in batch], dim=0),
            "attention_mask": torch.stack([item["attention_mask"] for item in batch], dim=0),
            "pixel_values": torch.stack([item["pixel_values"] for item in batch], dim=0),
            "social_features": torch.stack([item["social_features"] for item in batch], dim=0),
        }

        if "label" in batch[0]:
            collated["label"] = torch.stack([item["label"] for item in batch], dim=0)

        return collated


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
    
    Args:
        config: VMTFConfig settings.
        train_data: Filepath or DataFrame for training set.
        val_data: Filepath or DataFrame for validation set (optional).
        test_data: Filepath or DataFrame for test set (optional).
        tokenizer: Pre-instantiated HuggingFace Tokenizer (optional).
        image_processor: Pre-instantiated HuggingFace ImageProcessor (optional).
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader, fitted_social_normalizer)
    """
    # 1. Load HuggingFace Tokenizer and Image Processor once
    if tokenizer is None:
        from transformers import AutoTokenizer
        logger.info(f"Instantiating Tokenizer: {config.text_model_name}")
        tokenizer = AutoTokenizer.from_pretrained(config.text_model_name, use_fast=False)

    if image_processor is None:
        try:
            from transformers import AutoImageProcessor
            image_processor = AutoImageProcessor.from_pretrained(config.visual_model_name)
        except Exception as e:
            logger.warning(f"Could not load AutoImageProcessor: {e}. Using torchvision transforms.")
            image_processor = None

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

    collate_fn = VMTFCollate()

    # 3. Create Train DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        collate_fn=collate_fn,
        drop_last=len(train_dataset) > config.batch_size,
    )

    # 4. Create Validation DataLoader
    val_loader = None
    if val_data is not None:
        val_dataset = VMTFDataset(
            data=val_data,
            config=config,
            tokenizer=tokenizer,
            image_processor=image_processor,
            social_normalizer=social_normalizer,  # Reuse fitted normalizer
            is_train=False,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.val_batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            collate_fn=collate_fn,
            drop_last=False,
        )

    # 5. Create Test DataLoader
    test_loader = None
    test_dataset = None
    if test_data is not None:
        test_dataset = VMTFDataset(
            data=test_data,
            config=config,
            tokenizer=tokenizer,
            image_processor=image_processor,
            social_normalizer=social_normalizer,  # Reuse fitted normalizer
            is_train=False,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.val_batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            collate_fn=collate_fn,
            drop_last=False,
        )

    logger.info(
        f"DataLoaders successfully built: Train samples={len(train_dataset)}, "
        f"Val samples={len(val_dataset) if val_dataset is not None else 0}, "
        f"Test samples={len(test_dataset) if test_dataset is not None else 0}"
    )

    return train_loader, val_loader, test_loader, social_normalizer


# ----------------------------------------------------------------------
# Backward-Compatible & Master-Prompt Aliases
# ----------------------------------------------------------------------
VietnameseFakeNewsDataset = VMTFDataset
get_dataloaders = create_dataloaders
