"""
Text Encoder module for VMTF-Net using PhoBERT or ViSoBERT.
Extracts token sequence embeddings and [CLS] representation for Vietnamese text.
"""

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoModel

from ..config import VMTFConfig

logger = logging.getLogger(__name__)


class PhoBERTTextEncoder(nn.Module):
    """
    Vietnamese Text Encoder wrapping pretrained HuggingFace Transformer 
    (e.g., 'vinai/phobert-base' or 'UIT-NLP/visobert').
    
    Outputs:
        - token_embeddings: Sequence representation [batch_size, seq_len, text_dim]
        - cls_embedding: Pooled [CLS] sentence representation E_T in R^768 [batch_size, text_dim]
    """

    def __init__(self, config: VMTFConfig) -> None:
        super().__init__()
        self.config = config
        logger.info(f"Initializing Text Backbone: {config.text_model_name}")
        
        self.backbone = AutoModel.from_pretrained(config.text_model_name)
        
        # Optionally freeze bottom transformer layers for parameter-efficient fine-tuning
        if config.freeze_text_layers > 0:
            self._freeze_layers(config.freeze_text_layers)

    def _freeze_layers(self, num_layers_to_freeze: int) -> None:
        """Freeze embeddings and first N encoder layers."""
        logger.info(f"Freezing embeddings and first {num_layers_to_freeze} text transformer layers.")
        if hasattr(self.backbone, "embeddings"):
            for param in self.backbone.embeddings.parameters():
                param.requires_grad = False
                
        if hasattr(self.backbone, "encoder") and hasattr(self.backbone.encoder, "layer"):
            layers = self.backbone.encoder.layer
            for i in range(min(num_layers_to_freeze, len(layers))):
                for param in layers[i].parameters():
                    param.requires_grad = False

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through Vietnamese Text Encoder.
        
        Args:
            input_ids: Token indices of shape [batch_size, seq_len]
            attention_mask: Attention mask of shape [batch_size, seq_len]
            
        Returns:
            Tuple of:
                - token_embeddings: [batch_size, seq_len, text_dim (768)]
                - cls_embedding (E_T): [batch_size, text_dim (768)]
        """
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )

        # [batch_size, seq_len, hidden_dim]
        token_embeddings = outputs.last_hidden_state

        # Extract [CLS] token embedding (first token index 0)
        # PhoBERT/RoBERTa uses <s> at position 0
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            cls_embedding = outputs.pooler_output
        else:
            cls_embedding = token_embeddings[:, 0, :]

        return token_embeddings, cls_embedding
