"""
VMTF-Net: Vietnamese Multimodal Transformer Fusion Network
for Social Media Fake News Detection.
"""

from .config import VMTFConfig, BASIC_SOCIAL_COLS, EXTENDED_SOCIAL_COLS

try:
    from .dataset import (
        VMTFDataset,
        VMTFCollate,
        create_dataloaders,
        VietnameseFakeNewsDataset,
        get_dataloaders,
    )
    from .models import VMTFNet
except ImportError:
    VMTFDataset = None  # type: ignore
    VMTFCollate = None  # type: ignore
    create_dataloaders = None  # type: ignore
    VietnameseFakeNewsDataset = None  # type: ignore
    get_dataloaders = None  # type: ignore
    VMTFNet = None  # type: ignore

__all__ = [
    "VMTFConfig",
    "VMTFDataset",
    "VietnameseFakeNewsDataset",
    "VMTFCollate",
    "create_dataloaders",
    "get_dataloaders",
    "VMTFNet",
]

__version__ = "1.0.0"

