"""
Loss functions for VMTF-Net.
Implements:
1. Classification Loss: Standard Cross-Entropy or Focal Loss for class imbalance.
2. Text-Image Semantic Consistency Loss: Cosine similarity / contrastive alignment 
   penalizing semantic contradictions in out-of-context fake news.
3. VMTFCombinedLoss: Weighted combination L_total = L_cls + lambda * L_consistency.
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import VMTFConfig


class FocalLoss(nn.Module):
    """
    Multi-class / Binary Focal Loss to address class imbalance and hard examples:
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(
        self,
        gamma: float = 2.0,
        weight: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Predicted class logits [batch_size, num_classes]
            targets: Ground truth class indices [batch_size]
        """
        ce_loss = F.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


class TextImageConsistencyLoss(nn.Module):
    """
    Measures semantic alignment between Text ([CLS] token E_T) and Visual ([class] token E_V).
    
    Hypothesis for Multimodal Fake News:
    - Real News (label = 0): Text and image are coherent, complementary, and semantically aligned.
      Cosine similarity should be encouraged towards +1.
    - Fake News (label = 1): Often contains out-of-context images or contradictory claims.
      Distance / divergence is penalized or separated via margin.
    
    Loss formulation:
    L_align = 0.5 * (1 - cos_sim) for Real News (label == 0)
    L_mismatch = 0.5 * max(0, cos_sim - margin) for Fake News (label == 1)
    """

    def __init__(self, margin: float = 0.2, reduction: str = "mean") -> None:
        super().__init__()
        self.margin = margin
        self.reduction = reduction

    def forward(
        self,
        text_cls: torch.Tensor,
        visual_cls: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            text_cls: [batch_size, hidden_dim]
            visual_cls: [batch_size, hidden_dim]
            labels: [batch_size] (0: Real, 1: Fake). If None, computes standard alignment.
            
        Returns:
            Scalar consistency loss tensor.
        """
        # Normalize representations to unit hypersphere
        text_norm = F.normalize(text_cls, p=2, dim=-1)     # [B, D]
        visual_norm = F.normalize(visual_cls, p=2, dim=-1) # [B, D]
        
        # Cosine similarity in [-1, 1]
        cosine_sim = (text_norm * visual_norm).sum(dim=-1) # [B]

        if labels is None:
            # Unsupervised alignment: maximize cosine similarity
            loss = 1.0 - cosine_sim
        else:
            # Supervised alignment conditioned on truthfulness
            # Real news (label == 0): text and image should match -> min (1 - cos)
            loss_real = 1.0 - cosine_sim
            # Fake news (label == 1): out-of-context mismatch -> min max(0, cos - margin)
            loss_fake = F.relu(cosine_sim - self.margin)

            is_real = (labels == 0).float()
            is_fake = (labels == 1).float()

            loss = is_real * loss_real + is_fake * loss_fake

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class VMTFCombinedLoss(nn.Module):
    """
    Combined objective for VMTF-Net:
    Total_Loss = L_classification + lambda * L_consistency
    """

    def __init__(self, config: VMTFConfig, class_weights: Optional[torch.Tensor] = None) -> None:
        super().__init__()
        self.config = config
        self.consistency_lambda = config.consistency_lambda

        # 1. Classification Loss (CrossEntropy or Focal Loss)
        if config.use_focal_loss:
            self.cls_loss_fn = FocalLoss(gamma=config.focal_gamma, weight=class_weights)
        else:
            self.cls_loss_fn = nn.CrossEntropyLoss(weight=class_weights)

        # 2. Text-Image Consistency Loss
        self.consistency_loss_fn = TextImageConsistencyLoss(margin=0.2)

    def forward(
        self,
        model_outputs: Dict[str, torch.Tensor],
        targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined loss.
        
        Args:
            model_outputs: Dictionary from VMTFNet forward pass containing:
                - 'logits': [batch_size, num_classes]
                - 'text_cls': [batch_size, 768]
                - 'visual_cls': [batch_size, 768]
            targets: Ground truth class indices [batch_size]
            
        Returns:
            Tuple of:
                - total_loss: Scalar PyTorch tensor with gradients
                - loss_dict: Python dictionary with float loss metrics for logging
        """
        logits = model_outputs["logits"]
        text_cls = model_outputs["text_cls"]
        visual_cls = model_outputs["visual_cls"]

        # Classification Loss
        cls_loss = self.cls_loss_fn(logits, targets)

        # Consistency Loss
        if self.consistency_lambda > 0.0:
            consistency_loss = self.consistency_loss_fn(
                text_cls=text_cls,
                visual_cls=visual_cls,
                labels=targets,
            )
            total_loss = cls_loss + self.consistency_lambda * consistency_loss
        else:
            consistency_loss = torch.tensor(0.0, device=logits.device)
            total_loss = cls_loss

        loss_dict = {
            "total_loss": total_loss.item(),
            "cls_loss": cls_loss.item(),
            "consistency_loss": consistency_loss.item(),
        }

        return total_loss, loss_dict
