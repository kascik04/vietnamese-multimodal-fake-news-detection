"""
Text Encoder module for VMTF-Net using PhoBERT or ViSoBERT.
Extracts token sequence embeddings and a pooled sentence representation for Vietnamese text.
"""

import logging
from typing import Any, Tuple

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel

from ..config import VMTFConfig

logger = logging.getLogger(__name__)


def load_backbone(name: str, pretrained: bool = True) -> nn.Module:
    """
    Load a HuggingFace encoder without its pooler head.
    PhoBERT / ViSoBERT / ViT pooler weights are either absent from the checkpoint
    (randomly initialized) or unused, and unused parameters break DDP.

    Args:
        name: Hub id or local directory (use a local path for offline Kaggle notebooks).
        pretrained: If False, only the architecture is built (weights come from a VMTF checkpoint).
    """
    try:
        if pretrained:
            return AutoModel.from_pretrained(name, add_pooling_layer=False)
        return AutoModel.from_config(AutoConfig.from_pretrained(name), add_pooling_layer=False)
    except TypeError:
        # Architecture without `add_pooling_layer`: drop the pooler so checkpoints stay consistent
        model = AutoModel.from_pretrained(name) if pretrained else AutoModel.from_config(AutoConfig.from_pretrained(name))
        if getattr(model, "pooler", None) is not None:
            model.pooler = None
        return model


def enable_gradient_checkpointing(backbone: Any) -> None:
    """Enable non-reentrant gradient checkpointing when supported by the installed transformers."""
    try:
        backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    except TypeError:
        backbone.gradient_checkpointing_enable()


def freeze_bottom_layers(backbone: nn.Module, num_layers_to_freeze: int) -> None:
    """Freeze embeddings and the first N encoder layers of a BERT/ViT-style backbone."""
    if hasattr(backbone, "embeddings"):
        for param in backbone.embeddings.parameters():
            param.requires_grad = False
    if hasattr(backbone, "encoder") and hasattr(backbone.encoder, "layer"):
        layers = backbone.encoder.layer
        for i in range(min(num_layers_to_freeze, len(layers))):
            for param in layers[i].parameters():
                param.requires_grad = False


class PhoBERTTextEncoder(nn.Module):
    """
    Vietnamese Text Encoder wrapping pretrained HuggingFace Transformer
    (e.g., 'vinai/phobert-base' or 'uitnlp/visobert').

    Outputs:
        - token_embeddings: Sequence representation [batch_size, seq_len, text_dim]
        - sentence_embedding: E_T in R^768 [batch_size, text_dim] ('cls' or masked 'mean' pooling)
    """

    def __init__(self, config: VMTFConfig, pretrained: bool = True) -> None:
        super().__init__()
        self.config = config
        self.pooling = config.text_pooling
        logger.info(f"Initializing Text Backbone: {config.text_model_name} (pooling={self.pooling})")

        self.backbone = load_backbone(config.text_model_name, pretrained=pretrained)

        if config.freeze_text_layers > 0:
            logger.info(f"Freezing embeddings and first {config.freeze_text_layers} text transformer layers.")
            freeze_bottom_layers(self.backbone, config.freeze_text_layers)
        if config.gradient_checkpointing:
            enable_gradient_checkpointing(self.backbone)

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
                - sentence_embedding (E_T): [batch_size, text_dim (768)]
        """
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        token_embeddings = outputs.last_hidden_state  # [B, L, D]

        if self.pooling == "mean":
            mask = attention_mask.unsqueeze(-1).to(token_embeddings.dtype)
            sentence_embedding = (token_embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        else:
            # <s> token (RoBERTa-style CLS) at position 0
            sentence_embedding = token_embeddings[:, 0, :]

        return token_embeddings, sentence_embedding
