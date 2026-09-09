"""
Visual Encoder module for VMTF-Net using Vision Transformer (ViT).
Extracts patch token sequence embeddings and [class] representation for images.
"""

import logging
from typing import Tuple

import torch
import torch.nn as nn
from transformers import AutoModel

from ..config import VMTFConfig

logger = logging.getLogger(__name__)


class ViTVisualEncoder(nn.Module):
    """
    Visual Encoder wrapping pretrained HuggingFace Vision Transformer (ViT)
    (e.g., 'google/vit-base-patch16-224').
    
    Outputs:
        - patch_embeddings: Patch representations [batch_size, num_patches, visual_dim]
        - cls_embedding: Pooled [class] token representation E_V in R^768 [batch_size, visual_dim]
    """

    def __init__(self, config: VMTFConfig) -> None:
        super().__init__()
        self.config = config
        logger.info(f"Initializing Visual Backbone: {config.visual_model_name}")
        
        self.backbone = AutoModel.from_pretrained(config.visual_model_name)
        
        # Optionally freeze bottom visual transformer layers
        if config.freeze_visual_layers > 0:
            self._freeze_layers(config.freeze_visual_layers)

    def _freeze_layers(self, num_layers_to_freeze: int) -> None:
        """Freeze patch embeddings and first N encoder layers."""
        logger.info(f"Freezing embeddings and first {num_layers_to_freeze} ViT layers.")
        if hasattr(self.backbone, "embeddings"):
            for param in self.backbone.embeddings.parameters():
                param.requires_grad = False
                
        if hasattr(self.backbone, "encoder") and hasattr(self.backbone.encoder, "layer"):
            layers = self.backbone.encoder.layer
            for i in range(min(num_layers_to_freeze, len(layers))):
                for param in layers[i].parameters():
                    param.requires_grad = False

    def forward(self, pixel_values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through Vision Transformer.
        
        Args:
            pixel_values: Preprocessed image tensor of shape [batch_size, 3, 224, 224]
            
        Returns:
            Tuple of:
                - patch_embeddings: [batch_size, num_patches (196), visual_dim (768)]
                - cls_embedding (E_V): [batch_size, visual_dim (768)]
        """
        outputs = self.backbone(
            pixel_values=pixel_values,
            return_dict=True,
        )

        # [batch_size, num_patches + 1, hidden_dim]
        # In ViT-base-patch16-224: 14x14 = 196 patches + 1 [CLS] token = 197 tokens
        hidden_states = outputs.last_hidden_state

        # [CLS] token embedding at position 0
        cls_embedding = hidden_states[:, 0, :]  # [batch_size, 768]

        # Patch embeddings (all tokens after position 0)
        patch_embeddings = hidden_states[:, 1:, :]  # [batch_size, 196, 768]

        return patch_embeddings, cls_embedding
