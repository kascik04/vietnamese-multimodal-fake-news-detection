"""
VMTF-Net: Vietnamese Multimodal Transformer Fusion Network
for Social Media Fake News Detection.
"""

from .config import VMTFConfig, BASIC_SOCIAL_COLS, EXTENDED_SOCIAL_COLS, FUSION_TYPES

try:
    from .dataset import (
        SocialFeatureNormalizer,
        VMTFDataset,
        VMTFCollate,
        create_dataloaders,
        create_eval_loader,
        VietnameseFakeNewsDataset,
        get_dataloaders,
    )
    from .models import VMTFNet
except ImportError:
    SocialFeatureNormalizer = None  # type: ignore
    VMTFDataset = None  # type: ignore
    VMTFCollate = None  # type: ignore
    create_dataloaders = None  # type: ignore
    create_eval_loader = None  # type: ignore
    VietnameseFakeNewsDataset = None  # type: ignore
    get_dataloaders = None  # type: ignore
    VMTFNet = None  # type: ignore

__all__ = [
    "VMTFConfig",
    "FUSION_TYPES",
    "SocialFeatureNormalizer",
    "VMTFDataset",
    "VietnameseFakeNewsDataset",
    "VMTFCollate",
    "create_dataloaders",
    "create_eval_loader",
    "get_dataloaders",
    "VMTFNet",
]

__version__ = "2.0.0"

