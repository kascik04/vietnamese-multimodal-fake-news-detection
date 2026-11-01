"""
Multimodal Fusion module for VMTF-Net.
Implements:
1. Stage 1: Bidirectional Cross-Modal Attention between text tokens and visual patches,
   followed by a Contradiction module for Text-Image mismatch ("Out-of-Context Fakes").
2. Stage 2: Reliability-Aware Gated Fusion learning adaptive weights alpha + beta = 1.
3. Baselines for ablation studies: Late Concatenation and unimodal projections.

Every module accepts an optional `image_mask` [B] (1 = real image, 0 = missing image).
Missing images contribute nothing: cross-attention output is zeroed, beta is forced to 0
and the contradiction feature is zeroed, so the model cannot exploit placeholder pixels.
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import VMTFConfig


def _expand_mask(image_mask: Optional[torch.Tensor], ref: torch.Tensor) -> Optional[torch.Tensor]:
    """Reshape [B] image mask to broadcast against ref ([B, D] or [B, N, D])."""
    if image_mask is None:
        return None
    shape = [ref.size(0)] + [1] * (ref.dim() - 1)
    return image_mask.to(ref.dtype).view(*shape)


class CrossModalAttention(nn.Module):
    """
    Stage 1a: Bidirectional cross-modal attention modeling fine-grained token-to-patch and
    patch-to-token interactions.
    """

    def __init__(self, config: VMTFConfig) -> None:
        super().__init__()
        d_model = config.text_dim  # 768
        heads = config.cross_attn_heads
        dropout = config.cross_attn_dropout

        # 1. Text attends to Visual Patches (Query: Text, Key/Value: Visual)
        self.text_to_image_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            kdim=config.visual_dim,
            vdim=config.visual_dim,
            num_heads=heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ln_text = nn.LayerNorm(d_model)

        # 2. Visual Patches attend to Text (Query: Visual, Key/Value: Text)
        self.image_to_text_attn = nn.MultiheadAttention(
            embed_dim=config.visual_dim,
            kdim=d_model,
            vdim=d_model,
            num_heads=heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ln_image = nn.LayerNorm(config.visual_dim)

    def forward(
        self,
        text_tokens: torch.Tensor,
        patch_tokens: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            text_tokens: Text sequence tokens [batch_size, seq_len, 768]
            patch_tokens: Visual patch tokens [batch_size, num_patches, 768]
            attention_mask: Text padding mask [batch_size, seq_len] (1 for valid, 0 for pad)
            image_mask: [batch_size] (1 = image present)

        Returns:
            Tuple of:
                - attended_text_summary: [batch_size, 768]
                - attended_image_summary: [batch_size, 768] (zero for missing images)
        """
        # In PyTorch MultiheadAttention, key_padding_mask should be True for padded positions
        key_padding_mask = attention_mask == 0 if attention_mask is not None else None

        # A. Text attends to Visual Patches; no visual information is injected for missing images
        t2i_out, _ = self.text_to_image_attn(query=text_tokens, key=patch_tokens, value=patch_tokens)
        m3 = _expand_mask(image_mask, t2i_out)
        if m3 is not None:
            t2i_out = t2i_out * m3
        refined_text = self.ln_text(text_tokens + t2i_out)  # [B, L, 768]

        # Masked average pooling over valid text tokens -> [B, 768]
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).to(refined_text.dtype)
            text_summary = (refined_text * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1.0)
        else:
            text_summary = refined_text.mean(dim=1)

        # B. Visual Patches attend to Text Tokens
        i2t_out, _ = self.image_to_text_attn(
            query=patch_tokens,
            key=text_tokens,
            value=text_tokens,
            key_padding_mask=key_padding_mask,
        )
        refined_image = self.ln_image(patch_tokens + i2t_out)  # [B, P, 768]
        image_summary = refined_image.mean(dim=1)              # [B, 768]
        m2 = _expand_mask(image_mask, image_summary)
        if m2 is not None:
            image_summary = image_summary * m2

        return text_summary, image_summary


class ContradictionModule(nn.Module):
    """
    Stage 1b: Semantic discrepancy feature. Both modalities are first projected into a shared
    space (they come from different pre-trained encoders), then [|T - V| ; T * V] is encoded.
    """

    def __init__(self, config: VMTFConfig) -> None:
        super().__init__()
        d_model = config.text_dim
        self.out_dim = d_model // 2  # 384
        self.proj_text = nn.Linear(d_model, d_model)
        self.proj_visual = nn.Linear(config.visual_dim, d_model)
        self.contradiction_proj = nn.Sequential(
            nn.Linear(d_model * 2, self.out_dim),
            nn.LayerNorm(self.out_dim),
            nn.ReLU(),
            nn.Dropout(config.cross_attn_dropout),
        )

    def forward(
        self,
        text_repr: torch.Tensor,
        visual_repr: torch.Tensor,
        image_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            text_repr: [B, 768]; visual_repr: [B, 768]; image_mask: [B]
        Returns:
            contradiction_feat: [B, 384] (zero for missing images)
        """
        t = self.proj_text(text_repr)
        v = self.proj_visual(visual_repr)
        contra = self.contradiction_proj(torch.cat([torch.abs(t - v), t * v], dim=-1))
        m = _expand_mask(image_mask, contra)
        return contra * m if m is not None else contra


class CrossModalContradictionAttention(nn.Module):
    """
    Backward-compatible wrapper: cross-modal attention + contradiction module.
    Returns (text_summary, image_summary, contradiction_feat).
    """

    def __init__(self, config: VMTFConfig) -> None:
        super().__init__()
        self.attention = CrossModalAttention(config)
        self.contradiction = ContradictionModule(config)

    def forward(
        self,
        text_tokens: torch.Tensor,
        patch_tokens: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        text_summary, image_summary = self.attention(text_tokens, patch_tokens, attention_mask, image_mask)
        contradiction_feat = self.contradiction(text_summary, image_summary, image_mask)
        return text_summary, image_summary, contradiction_feat


class ReliabilityAwareGatedFusion(nn.Module):
    """
    Stage 2: Gating Mechanism learning dynamic modality reliability weights:
        alpha (text weight) and beta (visual weight) strictly satisfying:
        alpha + beta = 1, alpha >= 0, beta >= 0  (beta = 0 when the image is missing).

    Fuses gated text-visual representation with the Social Vector E_S and the
    contradiction feature (each optional for ablations).
    """

    def __init__(
        self,
        config: VMTFConfig,
        use_gate: bool = True,
        use_social: bool = True,
        use_contradiction: bool = True,
    ) -> None:
        super().__init__()
        d_model = config.text_dim  # 768
        dropout = config.gate_dropout
        self.use_gate = use_gate

        if use_gate:
            # Gating Network: [E_T ; E_V] -> 2 logits -> Softmax gives [alpha, beta]
            self.gate_net = nn.Sequential(
                nn.Linear(d_model + config.visual_dim, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, 2),
            )

        # gated_multimodal (768) [+ social (128)] [+ contradiction (384)]
        combined_dim = d_model
        if use_social:
            combined_dim += config.social_hidden_dim
        if use_contradiction:
            combined_dim += d_model // 2

        self.fusion_projection = nn.Sequential(
            nn.Linear(combined_dim, config.fusion_dim),
            nn.LayerNorm(config.fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        text_cls: torch.Tensor,
        visual_cls: torch.Tensor,
        social_feat: Optional[torch.Tensor] = None,
        contradiction_feat: Optional[torch.Tensor] = None,
        image_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            text_cls: Text representation E_T [batch_size, 768]
            visual_cls: Visual representation E_V [batch_size, 768]
            social_feat: Social engagement vector E_S [batch_size, 128] (optional)
            contradiction_feat: Discrepancy representation [batch_size, 384] (optional)
            image_mask: [batch_size] (1 = image present)

        Returns:
            Tuple of:
                - fused_representation: [batch_size, fusion_dim (768)]
                - gate_weights: Dict with 'alpha' [B, 1] and 'beta' [B, 1]
        """
        if self.use_gate:
            gate_logits = self.gate_net(torch.cat([text_cls, visual_cls], dim=-1))  # [B, 2]
            beta = F.softmax(gate_logits.float(), dim=-1)[:, 1:2].to(text_cls.dtype)
        else:
            beta = torch.full((text_cls.size(0), 1), 0.5, device=text_cls.device, dtype=text_cls.dtype)

        m = _expand_mask(image_mask, beta)
        if m is not None:
            beta = beta * m  # no image -> all weight on text
        alpha = 1.0 - beta   # alpha + beta = 1 by construction

        gated_multimodal = alpha * text_cls + beta * visual_cls  # [B, 768]

        parts = [gated_multimodal]
        if social_feat is not None:
            parts.append(social_feat)
        if contradiction_feat is not None:
            parts.append(contradiction_feat)
        fused_representation = self.fusion_projection(torch.cat(parts, dim=-1))

        return fused_representation, {"alpha": alpha, "beta": beta}


class LateConcatFusion(nn.Module):
    """
    Baseline Ablation Fusion: Simple concatenation of unimodal representations
    [E_T; E_V; E_S] followed by linear projection.
    """

    def __init__(self, config: VMTFConfig, use_social: bool = True) -> None:
        super().__init__()
        total_in = config.text_dim + config.visual_dim + (config.social_hidden_dim if use_social else 0)
        self.projection = nn.Sequential(
            nn.Linear(total_in, config.fusion_dim),
            nn.LayerNorm(config.fusion_dim),
            nn.ReLU(),
            nn.Dropout(config.classifier_dropout),
        )

    def forward(
        self,
        text_cls: torch.Tensor,
        visual_cls: torch.Tensor,
        social_feat: Optional[torch.Tensor] = None,
        image_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Returns:
            - fused_representation: [batch_size, fusion_dim (768)]
            - empty gate dict (no learned reliability weights in this baseline)
        """
        m = _expand_mask(image_mask, visual_cls)
        if m is not None:
            visual_cls = visual_cls * m
        parts = [text_cls, visual_cls] + ([social_feat] if social_feat is not None else [])
        return self.projection(torch.cat(parts, dim=-1)), {}


class UnimodalProjection(nn.Module):
    """Unimodal baseline (text_only / image_only / social_only): projection to fusion_dim."""

    def __init__(self, in_dim: int, config: VMTFConfig) -> None:
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(in_dim, config.fusion_dim),
            nn.LayerNorm(config.fusion_dim),
            nn.ReLU(),
            nn.Dropout(config.classifier_dropout),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        return self.projection(x), {}


class MultimodalFusion(nn.Module):
    """
    Unified Fusion Interface supporting ablation toggling via config:
    - fusion_type: 'hybrid_proposed' | 'late_concat' | 'text_only' | 'image_only' | 'social_only'
    - use_cross_attention / use_gate / use_contradiction / use_social (hybrid components)
    """

    def __init__(self, config: VMTFConfig) -> None:
        super().__init__()
        self.config = config
        self.fusion_type = config.fusion_type
        self.use_social = config.uses_social

        if self.fusion_type == "hybrid_proposed":
            self.use_cross_attention = config.use_cross_attention
            self.use_contradiction = config.use_contradiction
            if self.use_cross_attention:
                self.cross_attn = CrossModalAttention(config)
            if self.use_contradiction:
                self.contradiction = ContradictionModule(config)
            self.gated_fusion = ReliabilityAwareGatedFusion(
                config,
                use_gate=config.use_gate,
                use_social=self.use_social,
                use_contradiction=self.use_contradiction,
            )
        elif self.fusion_type == "late_concat":
            self.late_concat = LateConcatFusion(config, use_social=self.use_social)
        elif self.fusion_type == "text_only":
            self.unimodal = UnimodalProjection(config.text_dim, config)
        elif self.fusion_type == "image_only":
            self.unimodal = UnimodalProjection(config.visual_dim, config)
        elif self.fusion_type == "social_only":
            self.unimodal = UnimodalProjection(config.social_hidden_dim, config)
        else:
            raise ValueError(f"Unknown fusion_type '{self.fusion_type}'.")

    def forward(
        self,
        text_tokens: Optional[torch.Tensor] = None,
        text_cls: Optional[torch.Tensor] = None,
        patch_tokens: Optional[torch.Tensor] = None,
        visual_cls: Optional[torch.Tensor] = None,
        social_feat: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Unified fusion execution.

        Args:
            text_tokens: [B, L, 768]
            text_cls: [B, 768]
            patch_tokens: [B, P, 768]
            visual_cls: [B, 768]
            social_feat: [B, 128]
            attention_mask: [B, L]
            image_mask: [B]

        Returns:
            Tuple of:
                - fused_representation: [B, fusion_dim (768)]
                - gate_weights: Dict containing 'alpha' and 'beta' tensors (hybrid only)
        """
        if not self.use_social:
            social_feat = None

        if self.fusion_type == "text_only":
            return self.unimodal(text_cls)
        if self.fusion_type == "image_only":
            return self.unimodal(visual_cls)
        if self.fusion_type == "social_only":
            return self.unimodal(social_feat)
        if self.fusion_type == "late_concat":
            return self.late_concat(text_cls, visual_cls, social_feat, image_mask)

        # Proposed Hybrid Fusion
        # 1. Cross-Modal Attention -> enrich CLS embeddings with attended summaries
        if self.use_cross_attention:
            text_summary, image_summary = self.cross_attn(text_tokens, patch_tokens, attention_mask, image_mask)
            enriched_text = text_cls + text_summary
            enriched_visual = visual_cls + image_summary
        else:
            text_summary, image_summary = text_cls, visual_cls
            enriched_text, enriched_visual = text_cls, visual_cls

        # 2. Contradiction feature (on attended summaries, or on CLS when cross-attention is ablated)
        contradiction_feat = None
        if self.use_contradiction:
            contradiction_feat = self.contradiction(text_summary, image_summary, image_mask)

        # 3. Reliability-Aware Gated Fusion
        return self.gated_fusion(
            text_cls=enriched_text,
            visual_cls=enriched_visual,
            social_feat=social_feat,
            contradiction_feat=contradiction_feat,
            image_mask=image_mask,
        )
