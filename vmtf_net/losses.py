"""
Loss functions for VMTF-Net.
Implements:
1. Classification Loss: Standard Cross-Entropy or Focal Loss for class imbalance.
2. Text-Image Semantic Consistency Loss: cosine alignment in a learned SHARED space
   (projection heads), penalizing semantic contradictions in out-of-context fake news.
3. VMTFCombinedLoss: Weighted combination L_total = L_cls + lambda * L_consistency.
"""

from typing import Any, Dict, Optional, Tuple

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
        self.register_buffer("weight", weight if weight is not None else None)
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Predicted class logits [batch_size, num_classes]
            targets: Ground truth class indices [batch_size]
        """
        logits = logits.float()
        log_pt = F.log_softmax(logits, dim=-1).gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = log_pt.exp()
        focal_loss = -((1.0 - pt) ** self.gamma) * log_pt
        if self.weight is not None:
            focal_loss = focal_loss * self.weight[targets]

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


class TextImageConsistencyLoss(nn.Module):
    """
    Measures semantic alignment between projected text and image embeddings.

    Hypothesis for Multimodal Fake News:
    - Real News (label = 0): Text and image are coherent -> cosine similarity pushed towards +1.
    - Fake News (label = 1): Often out-of-context images -> cosine pushed below a margin.

    Loss formulation (only over samples that actually have an image):
    L_align    = 1 - cos_sim                 for Real News (label == 0)
    L_mismatch = max(0, cos_sim - margin)    for Fake News (label == 1)
    """

    def __init__(self, margin: float = 0.2, reduction: str = "mean") -> None:
        super().__init__()
        self.margin = margin
        self.reduction = reduction

    def forward(
        self,
        text_emb: torch.Tensor,
        visual_emb: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        image_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            text_emb: [batch_size, dim] (projected text embedding)
            visual_emb: [batch_size, dim] (projected visual embedding)
            labels: [batch_size] (0: Real, 1: Fake). If None, computes standard alignment.
            image_mask: [batch_size] (1 = image present). Missing-image samples are ignored.

        Returns:
            Scalar consistency loss tensor.
        """
        cosine_sim = F.cosine_similarity(text_emb.float(), visual_emb.float(), dim=-1)  # [B]

        if labels is None:
            loss = 1.0 - cosine_sim
        else:
            loss_real = 1.0 - cosine_sim
            loss_fake = F.relu(cosine_sim - self.margin)
            loss = torch.where(labels == 1, loss_fake, loss_real)

        if self.reduction == "none":
            return loss

        weights = image_mask.float() if image_mask is not None else torch.ones_like(loss)
        if self.reduction == "sum":
            return (loss * weights).sum()
        return (loss * weights).sum() / weights.sum().clamp(min=1.0)


class VMTFCombinedLoss(nn.Module):
    """
    Combined objective for VMTF-Net:
    Total_Loss = L_classification + lambda * L_consistency
    The consistency term is only active when the model exposes text/visual projections
    (multimodal experiments) and lambda > 0.
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
        self.consistency_loss_fn = TextImageConsistencyLoss(margin=config.consistency_margin)

    def forward(
        self,
        model_outputs: Dict[str, Any],
        targets: torch.Tensor,
        image_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined loss.

        Args:
            model_outputs: Dictionary from VMTFNet forward pass containing:
                - 'logits': [batch_size, num_classes]
                - 'text_proj' / 'visual_proj': [batch_size, proj_dim] (or None)
            targets: Ground truth class indices [batch_size]
            image_mask: [batch_size] (1 = image present)

        Returns:
            Tuple of:
                - total_loss: Scalar PyTorch tensor with gradients
                - loss_dict: Python dictionary with detached loss tensors for logging
        """
        logits = model_outputs["logits"].float()
        cls_loss = self.cls_loss_fn(logits, targets)

        text_proj = model_outputs.get("text_proj")
        visual_proj = model_outputs.get("visual_proj")
        if self.consistency_lambda > 0.0 and text_proj is not None and visual_proj is not None:
            consistency_loss = self.consistency_loss_fn(
                text_emb=text_proj,
                visual_emb=visual_proj,
                labels=targets,
                image_mask=image_mask,
            )
            total_loss = cls_loss + self.consistency_lambda * consistency_loss
        else:
            consistency_loss = torch.zeros((), device=logits.device)
            total_loss = cls_loss

        loss_dict = {
            "total_loss": total_loss.detach(),
            "cls_loss": cls_loss.detach(),
            "consistency_loss": consistency_loss.detach(),
        }

        return total_loss, loss_dict
