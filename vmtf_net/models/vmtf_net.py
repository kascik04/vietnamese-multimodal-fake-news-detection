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
from .social_encoder import SocialMetricsEncoder
from .text_encoder import PhoBERTTextEncoder
from .visual_encoder import ViTVisualEncoder


class ClassifierHead(nn.Module):
    """
    Multilayer classification head predicting binary real/fake news label.
    Architecture: Linear(in_features, hidden_dim) -> ReLU -> Dropout -> Linear(hidden_dim, num_classes)
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
    - Ablation support: Toggle between 'hybrid_proposed' and 'late_concat'.
    """

    def __init__(self, config: VMTFConfig) -> None:
        super().__init__()
        self.config = config

        # 1. Unimodal Encoders
        self.text_encoder = PhoBERTTextEncoder(config)
        self.visual_encoder = ViTVisualEncoder(config)
        self.social_encoder = SocialMetricsEncoder(config)

        # 2. Multimodal Fusion Module (supports 'hybrid_proposed' and 'late_concat')
        self.fusion = MultimodalFusion(config)

        # 3. Binary Classifier Head
        self.classifier = ClassifierHead(
            in_features=config.fusion_dim,
            hidden_dim=config.classifier_hidden_dim,
            num_classes=config.num_classes,
            dropout=config.classifier_dropout,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
        social_features: torch.Tensor,
    ) -> Dict[str, Any]:
        """
        Forward pass of VMTF-Net.
        
        Args:
            input_ids: Text token IDs [batch_size, seq_len]
            attention_mask: Text attention mask [batch_size, seq_len]
            pixel_values: Preprocessed image tensor [batch_size, 3, 224, 224]
            social_features: Engagement metrics [batch_size, social_input_dim]
            
        Returns:
            Dict containing:
                - 'logits': [batch_size, num_classes]
                - 'probs': [batch_size, num_classes]
                - 'text_cls': [batch_size, 768] (E_T for consistency loss)
                - 'visual_cls': [batch_size, 768] (E_V for consistency loss)
                - 'fused_feat': [batch_size, fusion_dim]
                - 'gate_weights': Dict with 'alpha' [B, 1] and 'beta' [B, 1]
                - 'fusion_type': str
        """
        # 1. Text Encoder Forward Pass
        # text_tokens: [B, L, 768], text_cls: [B, 768]
        text_tokens, text_cls = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # 2. Visual Encoder Forward Pass
        # patch_tokens: [B, P, 768], visual_cls: [B, 768]
        patch_tokens, visual_cls = self.visual_encoder(pixel_values=pixel_values)

        # 3. Social Metrics Encoder Forward Pass
        # social_feat: [B, 128]
        social_feat = self.social_encoder(social_features=social_features)

        # 4. Multimodal Fusion Forward Pass
        # fused_feat: [B, fusion_dim (768)]
        fused_feat, gate_weights = self.fusion(
            text_tokens=text_tokens,
            text_cls=text_cls,
            patch_tokens=patch_tokens,
            visual_cls=visual_cls,
            social_feat=social_feat,
            attention_mask=attention_mask,
        )

        # 5. Classifier Forward Pass
        # logits: [B, num_classes]
        logits = self.classifier(fused_feat)
        probs = F.softmax(logits, dim=-1)

        return {
            "logits": logits,
            "probs": probs,
            "text_cls": text_cls,
            "visual_cls": visual_cls,
            "fused_feat": fused_feat,
            "gate_weights": gate_weights,
            "fusion_type": self.config.fusion_type,
        }

    def get_parameter_groups(self) -> List[Dict[str, Any]]:
        """
        Organize parameters into groups with differential learning rates:
        - Backbone transformer layers fine-tune at learning_rate (e.g. 2e-5)
        - Newly initialized fusion, social MLP, and classifier heads train at head_learning_rate (e.g. 1e-4)
        - Excludes bias and LayerNorm/BatchNorm from weight decay to prevent underfitting.
        """
        no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight", "BatchNorm", "bn"]

        backbone_params_decay = []
        backbone_params_no_decay = []
        head_params_decay = []
        head_params_no_decay = []

        backbone_modules = [self.text_encoder.backbone, self.visual_encoder.backbone]
        head_modules = [
            self.social_encoder,
            self.fusion,
            self.classifier,
        ]

        # Backbone parameter grouping
        for module in backbone_modules:
            for name, param in module.named_parameters():
                if not param.requires_grad:
                    continue
                if any(nd in name for nd in no_decay):
                    backbone_params_no_decay.append(param)
                else:
                    backbone_params_decay.append(param)

        # Head parameter grouping
        for module in head_modules:
            for name, param in module.named_parameters():
                if not param.requires_grad:
                    continue
                if any(nd in name for nd in no_decay):
                    head_params_no_decay.append(param)
                else:
                    head_params_decay.append(param)

        return [
            {
                "params": backbone_params_decay,
                "lr": self.config.learning_rate,
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": backbone_params_no_decay,
                "lr": self.config.learning_rate,
                "weight_decay": 0.0,
            },
            {
                "params": head_params_decay,
                "lr": self.config.head_learning_rate,
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": head_params_no_decay,
                "lr": self.config.head_learning_rate,
                "weight_decay": 0.0,
            },
        ]
