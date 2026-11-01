"""
Full Integrated Model Architecture for VMTF-Net
(Vietnamese Multimodal Transformer Fusion Network).
"""

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import VMTFConfig
from .fusion import MultimodalFusion
from ..style_features import STYLE_FEATURE_NAMES
from .social_encoder import SocialMetricsEncoder, StyleFeatureEncoder
from .text_encoder import PhoBERTTextEncoder
from .visual_encoder import ViTVisualEncoder


class ClassifierHead(nn.Module):
    """
    Multilayer classification head predicting binary real/fake news label.
    Architecture: Linear(in_features, hidden_dim) -> LayerNorm -> ReLU -> Dropout -> Linear(hidden_dim, num_classes)
    """

    def __init__(self, in_features: int, hidden_dim: int, num_classes: int, dropout: float) -> None:
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Fused representation tensor [batch_size, in_features]
        Returns:
            Logits tensor [batch_size, num_classes]
        """
        return self.classifier(x)


class VMTFNet(nn.Module):
    """
    VMTF-Net: Vietnamese Multimodal Transformer Fusion Network.

    Processes three input modalities:
    1. Text: Vietnamese social post text -> PhoBERT/ViSoBERT -> E_T in R^768 & token sequence.
    2. Visual: Accompanying image -> ViT-base -> E_V in R^768 & patch sequence.
    3. Social: Engagement counts & reaction ratios -> MLP -> E_S in R^128.

    Fusion:
    - Stage 1: Bidirectional Cross-Modal Attention for contradiction detection.
    - Stage 2: Reliability-Aware Gated Fusion learning adaptive weights alpha + beta = 1.
    - Ablation support via config (fusion_type and use_* flags). Unused encoders are not built.

    The text-image consistency loss is computed in a shared space produced by two
    projection heads (PhoBERT and ViT embeddings live in unrelated spaces).
    """

    def __init__(self, config: VMTFConfig, pretrained_backbones: bool = True) -> None:
        """
        Args:
            config: VMTFConfig.
            pretrained_backbones: Load pretrained HF weights. Use False when the weights
                will be restored from a VMTF checkpoint (avoids a ~1GB download at evaluation).
        """
        super().__init__()
        self.config = config

        # 1. Unimodal Encoders (only those required by the experiment)
        self.text_encoder = PhoBERTTextEncoder(config, pretrained_backbones) if config.uses_text else None
        self.visual_encoder = ViTVisualEncoder(config, pretrained_backbones) if config.uses_image else None
        self.social_encoder = SocialMetricsEncoder(config) if config.uses_social else None
        self.style_encoder = (
            StyleFeatureEncoder(config, in_dim=len(STYLE_FEATURE_NAMES)) if config.uses_style else None
        )

        # 2. Shared space for the consistency loss / cosine interpretability
        # (only built when the loss trains it: untrained heads would report a meaningless cosine)
        self.has_consistency_heads = config.uses_text and config.uses_image and config.consistency_lambda > 0
        if self.has_consistency_heads:
            self.consistency_text_proj = nn.Linear(config.text_dim, config.consistency_proj_dim)
            self.consistency_visual_proj = nn.Linear(config.visual_dim, config.consistency_proj_dim)

        # 3. Multimodal Fusion Module
        self.fusion = MultimodalFusion(config)

        # 4. Binary Classifier Head
        self.classifier = ClassifierHead(
            in_features=config.fusion_dim,
            hidden_dim=config.classifier_hidden_dim,
            num_classes=config.num_classes,
            dropout=config.classifier_dropout,
        )

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        social_features: Optional[torch.Tensor] = None,
        image_mask: Optional[torch.Tensor] = None,
        style_features: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Forward pass of VMTF-Net.

        Args:
            input_ids: Text token IDs [batch_size, seq_len]
            attention_mask: Text attention mask [batch_size, seq_len]
            pixel_values: Preprocessed image tensor [batch_size, 3, 224, 224]
            social_features: Engagement metrics [batch_size, social_input_dim]
            image_mask: [batch_size] float, 1 = image present, 0 = missing (optional)

        Returns:
            Dict containing (keys absent from the experiment are None):
                - 'logits': [batch_size, num_classes]
                - 'probs': [batch_size, num_classes]
                - 'text_cls': [batch_size, 768]
                - 'visual_cls': [batch_size, 768]
                - 'text_proj' / 'visual_proj': [batch_size, consistency_proj_dim]
                - 'cosine_sim': [batch_size] text-image cosine in the shared space
                - 'fused_feat': [batch_size, fusion_dim]
                - 'gate_weights': Dict with 'alpha' [B, 1] and 'beta' [B, 1] (hybrid only)
        """
        text_tokens = text_cls = patch_tokens = visual_cls = social_feat = None

        if self.text_encoder is not None:
            text_tokens, text_cls = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)

        if self.visual_encoder is not None:
            patch_tokens, visual_cls = self.visual_encoder(pixel_values=pixel_values)

        if self.social_encoder is not None:
            social_feat = self.social_encoder(social_features=social_features)

        style_feat = None
        if self.style_encoder is not None:
            if style_features is None:
                raise ValueError("Model was built with use_style_features=True but no style_features were given.")
            style_feat = self.style_encoder(style_features)

        fused_feat, gate_weights = self.fusion(
            text_tokens=text_tokens,
            text_cls=text_cls,
            patch_tokens=patch_tokens,
            visual_cls=visual_cls,
            social_feat=social_feat,
            attention_mask=attention_mask,
            image_mask=image_mask,
            style_feat=style_feat,
        )

        logits = self.classifier(fused_feat)
        probs = F.softmax(logits.float(), dim=-1)

        text_proj = visual_proj = cosine_sim = None
        if self.has_consistency_heads:
            text_proj = self.consistency_text_proj(text_cls)
            visual_proj = self.consistency_visual_proj(visual_cls)
            cosine_sim = F.cosine_similarity(text_proj.float(), visual_proj.float(), dim=-1)

        return {
            "logits": logits,
            "probs": probs,
            "text_cls": text_cls,
            "visual_cls": visual_cls,
            "text_proj": text_proj,
            "visual_proj": visual_proj,
            "cosine_sim": cosine_sim,
            "fused_feat": fused_feat,
            "gate_weights": gate_weights,
        }

    def count_parameters(self) -> Dict[str, int]:
        """Total / trainable parameter counts (reported in run_info.json)."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}

    def get_parameter_groups(self) -> List[Dict[str, Any]]:
        """
        Organize parameters into groups with differential learning rates:
        - Backbone transformer layers fine-tune at learning_rate (e.g. 2e-5)
        - Newly initialized fusion, social MLP, projection and classifier heads train at head_learning_rate
        - Biases and all normalization weights (1-D tensors) are excluded from weight decay.
        """
        backbone_ids = set()
        for encoder in (self.text_encoder, self.visual_encoder):
            if encoder is not None:
                backbone_ids.update(id(p) for p in encoder.backbone.parameters())

        groups = {
            ("backbone", True): [],
            ("backbone", False): [],
            ("head", True): [],
            ("head", False): [],
        }
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            part = "backbone" if id(param) in backbone_ids else "head"
            decay = param.ndim >= 2 and not name.endswith(".bias")
            groups[(part, decay)].append(param)

        lr = {"backbone": self.config.learning_rate, "head": self.config.head_learning_rate}
        return [
            {
                "params": params,
                "lr": lr[part],
                "weight_decay": self.config.weight_decay if decay else 0.0,
            }
            for (part, decay), params in groups.items()
            if params
        ]
