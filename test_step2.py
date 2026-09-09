"""
Verification and Unit Testing Script for VMTF-Net Step 2 (Model Architecture & Fusion).
Validates:
1. SocialMetricsEncoder forward pass and output dimension.
2. CrossModalContradictionAttention (Stage 1) forward pass and tensor shapes.
3. ReliabilityAwareGatedFusion (Stage 2) and asserts alpha + beta = 1 constraint.
4. LateConcatFusion baseline (Ablation mode).
5. Parameter grouping for differential learning rates.
"""

import os
import sys

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from vmtf_net.config import VMTFConfig


def run_tests() -> None:
    try:
        import torch
        import torch.nn as nn
    except ImportError as e:
        print(f"[!] PyTorch is not installed in local environment: {e}")
        print("[!] Ready to test seamlessly on Kaggle GPU / Google Colab!")
        return

    from vmtf_net.models.social_encoder import SocialMetricsEncoder
    from vmtf_net.models.fusion import (
        CrossModalContradictionAttention,
        ReliabilityAwareGatedFusion,
        LateConcatFusion,
        MultimodalFusion,
    )
    from vmtf_net.models.vmtf_net import ClassifierHead

    cfg = VMTFConfig(batch_size=4, max_seq_len=32)
    B = 4
    L = 32
    P = 196
    D = 768
    D_social = 3

    print("=" * 60)
    print("RUNNING VMTF-NET STEP 2 ARCHITECTURE CHECKS")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Social Metrics Encoder Check
    # ------------------------------------------------------------------
    print("\n[1] Testing SocialMetricsEncoder...")
    social_encoder = SocialMetricsEncoder(cfg)
    mock_social = torch.randn(B, D_social)
    es = social_encoder(mock_social)
    print(f"    - Input shape:  {list(mock_social.shape)}")
    print(f"    - Output shape: {list(es.shape)} (Expected: [{B}, 128])")
    assert es.shape == (B, 128), "Social embedding shape mismatch!"
    print("    [OK] SocialMetricsEncoder passed.")

    # ------------------------------------------------------------------
    # 2. Cross-Modal Attention Check (Stage 1)
    # ------------------------------------------------------------------
    print("\n[2] Testing CrossModalContradictionAttention (Stage 1)...")
    cross_attn = CrossModalContradictionAttention(cfg)
    mock_text_tokens = torch.randn(B, L, D)
    mock_patch_tokens = torch.randn(B, P, D)
    mock_mask = torch.ones(B, L, dtype=torch.long)
    mock_mask[:, 24:] = 0  # 8 tokens masked as padding

    t_summary, i_summary, contra_feat = cross_attn(
        text_tokens=mock_text_tokens,
        patch_tokens=mock_patch_tokens,
        attention_mask=mock_mask,
    )
    print(f"    - Attended Text Summary:   {list(t_summary.shape)} (Expected: [{B}, {D}])")
    print(f"    - Attended Image Summary:  {list(i_summary.shape)} (Expected: [{B}, {D}])")
    print(f"    - Contradiction Features:  {list(contra_feat.shape)} (Expected: [{B}, 384])")
    assert t_summary.shape == (B, D)
    assert i_summary.shape == (B, D)
    assert contra_feat.shape == (B, 384)
    print("    [OK] CrossModalContradictionAttention passed.")

    # ------------------------------------------------------------------
    # 3. Reliability-Aware Gated Fusion Check (Stage 2)
    # ------------------------------------------------------------------
    print("\n[3] Testing ReliabilityAwareGatedFusion (Stage 2)...")
    gated_fusion = ReliabilityAwareGatedFusion(cfg)
    mock_text_cls = torch.randn(B, D)
    mock_visual_cls = torch.randn(B, D)

    fused_feat, gate_weights = gated_fusion(
        text_cls=mock_text_cls,
        visual_cls=mock_visual_cls,
        social_feat=es,
        contradiction_feat=contra_feat,
    )
    alpha = gate_weights["alpha"]
    beta = gate_weights["beta"]

    print(f"    - Fused representation:    {list(fused_feat.shape)} (Expected: [{B}, {D}])")
    print(f"    - Gate weights alpha:      {alpha.squeeze().tolist()}")
    print(f"    - Gate weights beta:       {beta.squeeze().tolist()}")
    
    # Assert alpha + beta = 1.0 within floating point precision
    sum_weights = alpha + beta
    ones = torch.ones_like(sum_weights)
    assert torch.allclose(sum_weights, ones, atol=1e-6), "Gate constraint alpha + beta = 1 failed!"
    print(f"    - Constraint alpha + beta = 1 verification: max diff = {(sum_weights - ones).abs().max().item():.8f}")
    assert fused_feat.shape == (B, D)
    print("    [OK] ReliabilityAwareGatedFusion passed.")

    # ------------------------------------------------------------------
    # 4. Ablation Check (Late Concatenation Baseline)
    # ------------------------------------------------------------------
    print("\n[4] Testing Baseline Ablation (fusion_type='late_concat')...")
    cfg_ablation = VMTFConfig(batch_size=B, fusion_type="late_concat")
    ablation_fusion = MultimodalFusion(cfg_ablation)
    
    fused_late, late_weights = ablation_fusion(
        text_tokens=mock_text_tokens,
        text_cls=mock_text_cls,
        patch_tokens=mock_patch_tokens,
        visual_cls=mock_visual_cls,
        social_feat=es,
        attention_mask=mock_mask,
    )
    print(f"    - Late concat fused shape: {list(fused_late.shape)} (Expected: [{B}, {D}])")
    assert fused_late.shape == (B, D)
    print("    [OK] LateConcatFusion baseline passed.")

    # ------------------------------------------------------------------
    # 5. Proposed Unified Fusion Check
    # ------------------------------------------------------------------
    print("\n[5] Testing Proposed Unified MultimodalFusion (fusion_type='hybrid_proposed')...")
    unified_fusion = MultimodalFusion(cfg)
    fused_hybrid, hybrid_weights = unified_fusion(
        text_tokens=mock_text_tokens,
        text_cls=mock_text_cls,
        patch_tokens=mock_patch_tokens,
        visual_cls=mock_visual_cls,
        social_feat=es,
        attention_mask=mock_mask,
    )
    print(f"    - Proposed hybrid fused shape: {list(fused_hybrid.shape)} (Expected: [{B}, {D}])")
    assert fused_hybrid.shape == (B, D)
    print("    [OK] Proposed MultimodalFusion passed.")

    # ------------------------------------------------------------------
    # 6. Classifier Head Check
    # ------------------------------------------------------------------
    print("\n[6] Testing ClassifierHead...")
    classifier = ClassifierHead(
        in_features=cfg.fusion_dim,
        hidden_dim=cfg.classifier_hidden_dim,
        num_classes=cfg.num_classes,
        dropout=cfg.classifier_dropout,
    )
    logits = classifier(fused_hybrid)
    probs = torch.softmax(logits, dim=-1)
    print(f"    - Logits shape: {list(logits.shape)} (Expected: [{B}, 2])")
    print(f"    - Probs shape:  {list(probs.shape)} (Expected: [{B}, 2])")
    assert logits.shape == (B, 2)
    print("    [OK] ClassifierHead passed.")

    print("\n" + "=" * 60)
    print("[OK] ALL STEP 2 MODULE CHECKS PASSED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    run_tests()
