"""
Multimodal Fusion module for VMTF-Net.
Implements:
1. Stage 1: Bidirectional Cross-Modal Attention between text tokens and visual patches 
   for Text-Image Contradiction Detection ("Out-of-Context Fakes").
2. Stage 2: Reliability-Aware Gated Fusion learning adaptive weights alpha + beta = 1.
3. Baseline Late Concatenation for ablation studies.
"""

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import VMTFConfig


class CrossModalContradictionAttention(nn.Module):
    """
    Stage 1: Cross-Modal Attention modeling fine-grained token-to-patch and
    patch-to-token interactions to identify semantic discrepancies and out-of-context fakes.
    """

    def __init__(self, config: VMTFConfig) -> None:
        super().__init__()
        self.config = config
        d_model = config.text_dim  # 768
        heads = config.cross_attn_heads
        dropout = config.cross_attn_dropout

        # 1. Text attends to Visual Patches (Query: Text, Key/Value: Visual)
        self.text_to_image_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ln_text = nn.LayerNorm(d_model)

        # 2. Visual Patches attend to Text (Query: Visual, Key/Value: Text)
        self.image_to_text_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ln_image = nn.LayerNorm(d_model)

        # 3. Contradiction Projection: maps difference & product vectors to dense discrepancy feature
        # Difference |T - V| (768) + Element-wise Product T * V (768) = 1536
        self.contradiction_proj = nn.Sequential(
            nn.Linear(d_model * 2, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        text_tokens: torch.Tensor,
        patch_tokens: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for Cross-Modal Attention.
        
        Args:
            text_tokens: Text sequence tokens [batch_size, seq_len, 768]
            patch_tokens: Visual patch tokens [batch_size, num_patches, 768]
            attention_mask: Text padding mask [batch_size, seq_len] (1 for valid, 0 for pad)
            
        Returns:
            Tuple of:
                - attended_text_summary: [batch_size, 768]
                - attended_image_summary: [batch_size, 768]
                - contradiction_feat: [batch_size, 384]
        """
        batch_size = text_tokens.size(0)

        # In PyTorch MultiheadAttention, key_padding_mask should be True for padded positions
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = attention_mask == 0  # [batch_size, seq_len]

        # ------------------------------------------------------------------
        # A. Text attends to Visual Patches
        # ------------------------------------------------------------------
        # Query: text_tokens [B, L, 768], Key/Value: patch_tokens [B, P, 768]
        t2i_out, _ = self.text_to_image_attn(
            query=text_tokens,
            key=patch_tokens,
            value=patch_tokens,
        )
        # Residual + LayerNorm: [B, L, 768]
        refined_text = self.ln_text(text_tokens + t2i_out)

        # Masked average pooling over valid text tokens -> [B, 768]
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).float()  # [B, L, 1]
            sum_text = (refined_text * mask_expanded).sum(dim=1)  # [B, 768]
            lengths = mask_expanded.sum(dim=1).clamp(min=1e-9)   # [B, 1]
            text_summary = sum_text / lengths
        else:
            text_summary = refined_text.mean(dim=1)

        # ------------------------------------------------------------------
        # B. Visual Patches attend to Text Tokens
        # ------------------------------------------------------------------
        # Query: patch_tokens [B, P, 768], Key/Value: text_tokens [B, L, 768]
        i2t_out, _ = self.image_to_text_attn(
            query=patch_tokens,
            key=text_tokens,
            value=text_tokens,
            key_padding_mask=key_padding_mask,
        )
        # Residual + LayerNorm: [B, P, 768]
        refined_image = self.ln_image(patch_tokens + i2t_out)
        image_summary = refined_image.mean(dim=1)  # [B, 768]

        # ------------------------------------------------------------------
        # C. Contradiction & Semantic Discrepancy Modeling
        # ------------------------------------------------------------------
        # Absolute difference and element-wise product to detect mismatch
        diff_feat = torch.abs(text_summary - image_summary)  # [B, 768]
        prod_feat = text_summary * image_summary             # [B, 768]
        contra_combined = torch.cat([diff_feat, prod_feat], dim=-1)  # [B, 1536]
        contradiction_feat = self.contradiction_proj(contra_combined)  # [B, 384]

        return text_summary, image_summary, contradiction_feat


class ReliabilityAwareGatedFusion(nn.Module):
    """
    Stage 2: Gating Mechanism learning dynamic modality reliability weights:
        alpha (text weight) and beta (visual weight) strictly satisfying:
        alpha + beta = 1, alpha >= 0, beta >= 0.
        
    Fuses gated text-visual representation with the Social Vector E_S.
    """

    def __init__(self, config: VMTFConfig) -> None:
        super().__init__()
        self.config = config
        d_model = config.text_dim  # 768
        d_social = config.social_hidden_dim  # 128
        d_contra = d_model // 2  # 384
        fusion_dim = config.fusion_dim  # 768
        dropout = config.gate_dropout

        # Gating Network: takes concatenated text and visual representations
        # and outputs 2 logits [batch_size, 2] -> Softmax gives [alpha, beta]
        self.gate_net = nn.Sequential(
            nn.Linear(d_model * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 2),
        )

        # Total fused dimension:
        # gated_multimodal (768) + social_feat (128) + contradiction_feat (384) = 1280
        combined_dim = d_model + d_social + d_contra

        self.fusion_projection = nn.Sequential(
            nn.Linear(combined_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        text_cls: torch.Tensor,
        visual_cls: torch.Tensor,
        social_feat: torch.Tensor,
        contradiction_feat: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass for Reliability-Aware Gated Fusion.
        
        Args:
            text_cls: Pooled text embedding E_T [batch_size, 768]
            visual_cls: Pooled visual embedding E_V [batch_size, 768]
            social_feat: Social engagement vector E_S [batch_size, 128]
            contradiction_feat: Discrepancy representation [batch_size, 384]
            
        Returns:
            Tuple of:
                - fused_representation: [batch_size, fusion_dim (768)]
                - gate_weights: Dict with 'alpha' [B, 1] and 'beta' [B, 1]
        """
        # 1. Compute dynamic reliability weights
        gate_input = torch.cat([text_cls, visual_cls], dim=-1)  # [B, 1536]
        gate_logits = self.gate_net(gate_input)                  # [B, 2]
        
        # Softmax ensures alpha + beta = 1
        gate_weights = F.softmax(gate_logits, dim=-1)            # [B, 2]
        alpha = gate_weights[:, 0:1]                             # [B, 1] (Text reliability)
        beta = gate_weights[:, 1:2]                              # [B, 1] (Visual reliability)

        # 2. Convex combination of text and image representations
        gated_multimodal = alpha * text_cls + beta * visual_cls  # [B, 768]

        # 3. Concatenate with Social Vector E_S and Contradiction Features
        fused_cat = torch.cat([gated_multimodal, social_feat, contradiction_feat], dim=-1)  # [B, 1280]
        
        # 4. Final projection
        fused_representation = self.fusion_projection(fused_cat)  # [B, fusion_dim (768)]

        return fused_representation, {"alpha": alpha, "beta": beta}


class LateConcatFusion(nn.Module):
    """
    Baseline Ablation Fusion: Simple concatenation of unimodal representations
    [E_T; E_V; E_S] followed by linear projection.
    Corresponds to standard baseline methods (e.g., RoViT-Detect).
    """

    def __init__(self, config: VMTFConfig) -> None:
        super().__init__()
        self.config = config
        d_text = config.text_dim  # 768
        d_visual = config.visual_dim  # 768
        d_social = config.social_hidden_dim  # 128
        fusion_dim = config.fusion_dim  # 768
        dropout = config.classifier_dropout

        total_in = d_text + d_visual + d_social  # 1664

        self.projection = nn.Sequential(
            nn.Linear(total_in, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        text_cls: torch.Tensor,
        visual_cls: torch.Tensor,
        social_feat: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass for baseline concatenation.
        
        Args:
            text_cls: [batch_size, 768]
            visual_cls: [batch_size, 768]
            social_feat: [batch_size, 128]
            
        Returns:
            Tuple of:
                - fused_representation: [batch_size, fusion_dim (768)]
                - dummy_weights: Dict with uniform 0.5 weights for consistent logging
        """
        batch_size = text_cls.size(0)
        device = text_cls.device
        
        concat_feat = torch.cat([text_cls, visual_cls, social_feat], dim=-1)  # [B, 1664]
        fused_representation = self.projection(concat_feat)                   # [B, 768]

        # Uniform weights for baseline reporting
        dummy_alpha = torch.full((batch_size, 1), 0.5, device=device)
        dummy_beta = torch.full((batch_size, 1), 0.5, device=device)

        return fused_representation, {"alpha": dummy_alpha, "beta": dummy_beta}


class MultimodalFusion(nn.Module):
    """
    Unified Fusion Interface supporting ablation toggling via config.fusion_type:
    - 'hybrid_proposed': Cross-Attention (Contradiction) + Reliability-Aware Gated Fusion.
    - 'late_concat': Simple Late Concatenation baseline.
    """

    def __init__(self, config: VMTFConfig) -> None:
        super().__init__()
        self.config = config
        self.fusion_type = config.fusion_type

        if self.fusion_type == "hybrid_proposed":
            self.cross_attn = CrossModalContradictionAttention(config)
            self.gated_fusion = ReliabilityAwareGatedFusion(config)
        elif self.fusion_type == "late_concat":
            self.late_concat = LateConcatFusion(config)
        else:
            raise ValueError(f"Unknown fusion_type '{self.fusion_type}'. Choose 'hybrid_proposed' or 'late_concat'.")

    def forward(
        self,
        text_tokens: torch.Tensor,
        text_cls: torch.Tensor,
        patch_tokens: torch.Tensor,
        visual_cls: torch.Tensor,
        social_feat: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
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
            
        Returns:
            Tuple of:
                - fused_representation: [B, fusion_dim (768)]
                - gate_weights: Dict containing 'alpha' and 'beta' tensors
        """
        if self.fusion_type == "late_concat":
            return self.late_concat(text_cls, visual_cls, social_feat)

        # Proposed Hybrid Fusion:
        # 1. Cross-Modal Attention & Contradiction Detection
        text_summary, image_summary, contradiction_feat = self.cross_attn(
            text_tokens=text_tokens,
            patch_tokens=patch_tokens,
            attention_mask=attention_mask,
        )

        # 2. Enrich CLS embeddings with attended summaries
        enriched_text = text_cls + text_summary
        enriched_visual = visual_cls + image_summary

        # 3. Reliability-Aware Gated Fusion
        fused_feat, gate_weights = self.gated_fusion(
            text_cls=enriched_text,
            visual_cls=enriched_visual,
            social_feat=social_feat,
            contradiction_feat=contradiction_feat,
        )

        return fused_feat, gate_weights
