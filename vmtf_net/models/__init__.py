"""
Models package for VMTF-Net.
"""

from .text_encoder import PhoBERTTextEncoder
from .visual_encoder import ViTVisualEncoder
from .social_encoder import SocialMetricsEncoder
from .fusion import (
    ContradictionModule,
    CrossModalAttention,
    CrossModalContradictionAttention,
    ReliabilityAwareGatedFusion,
    LateConcatFusion,
    UnimodalProjection,
    MultimodalFusion,
)
from .vmtf_net import VMTFNet, ClassifierHead

__all__ = [
    "PhoBERTTextEncoder",
    "ViTVisualEncoder",
    "SocialMetricsEncoder",
    "ContradictionModule",
    "CrossModalAttention",
    "CrossModalContradictionAttention",
    "ReliabilityAwareGatedFusion",
    "LateConcatFusion",
    "UnimodalProjection",
    "MultimodalFusion",
    "VMTFNet",
    "ClassifierHead",
]
