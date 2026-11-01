"""
Verification and Unit Testing Script for VMTF-Net Step 2 (Model Architecture & Fusion) - OFFLINE.
Validates:
1. SocialMetricsEncoder forward pass and output dimension.
2. CrossModalAttention + ContradictionModule (Stage 1) shapes and missing-image masking.
3. ReliabilityAwareGatedFusion (Stage 2): alpha + beta = 1 and beta = 0 without image.
4. LateConcatFusion and unimodal baselines.
5. Every hybrid ablation variant (no cross-attention / gate / contradiction / social).
"""

import os
import sys
import tempfile

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
    except ImportError as e:
        print(f"[!] PyTorch is not installed in local environment: {e}")
        return

    from vmtf_net.models.fusion import (
        ContradictionModule,
        CrossModalAttention,
        CrossModalContradictionAttention,
        LateConcatFusion,
        MultimodalFusion,
        ReliabilityAwareGatedFusion,
    )
    from vmtf_net.models.social_encoder import SocialMetricsEncoder
    from vmtf_net.models.vmtf_net import ClassifierHead

    tmp = tempfile.mkdtemp()
    common = dict(checkpoint_dir=os.path.join(tmp, "c"), log_dir=os.path.join(tmp, "l"), device="cpu")
    cfg = VMTFConfig(batch_size=4, max_seq_len=32, **common)
    B, L, P, D, D_social = 4, 32, 196, 768, 3
    torch.manual_seed(0)

    print("=" * 60)
    print("RUNNING VMTF-NET STEP 2 ARCHITECTURE CHECKS")
    print("=" * 60)

    # 1. Social Metrics Encoder
    print("\n[1] Testing SocialMetricsEncoder...")
    social_encoder = SocialMetricsEncoder(cfg)
    es = social_encoder(torch.randn(B, D_social))
    assert es.shape == (B, 128), "Social embedding shape mismatch!"
    assert social_encoder(torch.randn(1, D_social)).shape == (1, 128), "Batch of 1 must work in train mode"
    print("    [OK] SocialMetricsEncoder passed.")

    # 2. Cross-Modal Attention + Contradiction
    print("\n[2] Testing CrossModalAttention + ContradictionModule (Stage 1)...")
    text_tokens = torch.randn(B, L, D)
    patch_tokens = torch.randn(B, P, D)
    mask = torch.ones(B, L, dtype=torch.long)
    mask[:, 24:] = 0
    image_mask = torch.tensor([1.0, 1.0, 0.0, 1.0])

    attn = CrossModalAttention(cfg).eval()
    t_sum, i_sum = attn(text_tokens, patch_tokens, mask, image_mask)
    assert t_sum.shape == (B, D) and i_sum.shape == (B, D)
    assert torch.all(i_sum[2] == 0), "Image summary must be zero for a missing image"

    # Text summary of a no-image sample must not depend on the (placeholder) patches
    t_sum2, _ = attn(text_tokens, torch.randn(B, P, D), mask, image_mask)
    assert torch.allclose(t_sum[2], t_sum2[2], atol=1e-5), "Missing image leaked into text summary"

    contra_mod = ContradictionModule(cfg).eval()
    contra = contra_mod(t_sum, i_sum, image_mask)
    assert contra.shape == (B, 384) and torch.all(contra[2] == 0)

    _, _, contra_compat = CrossModalContradictionAttention(cfg)(text_tokens, patch_tokens, mask, image_mask)
    assert contra_compat.shape == (B, 384)
    print("    [OK] Stage 1 shapes and missing-image masking passed.")

    # 3. Gated fusion
    print("\n[3] Testing ReliabilityAwareGatedFusion (Stage 2)...")
    gated = ReliabilityAwareGatedFusion(cfg)
    fused, w = gated(torch.randn(B, D), torch.randn(B, D), es, contra, image_mask)
    alpha, beta = w["alpha"], w["beta"]
    print(f"    - alpha: {[round(x, 3) for x in alpha.squeeze(-1).tolist()]}")
    print(f"    - beta:  {[round(x, 3) for x in beta.squeeze(-1).tolist()]}")
    assert torch.allclose(alpha + beta, torch.ones_like(alpha), atol=1e-6), "alpha + beta = 1 failed!"
    assert beta[2].item() == 0.0 and alpha[2].item() == 1.0, "beta must be 0 when the image is missing"
    assert fused.shape == (B, D)
    print("    [OK] ReliabilityAwareGatedFusion passed.")

    # 4. Late concat baseline
    print("\n[4] Testing LateConcatFusion...")
    late = LateConcatFusion(cfg)
    fused_late, late_w = late(torch.randn(B, D), torch.randn(B, D), es, image_mask)
    assert fused_late.shape == (B, D) and late_w == {}
    print("    [OK] LateConcatFusion passed.")

    # 5. Every fusion type / ablation through the unified interface
    print("\n[5] Testing MultimodalFusion for every experiment configuration...")
    text_cls, visual_cls = torch.randn(B, D), torch.randn(B, D)
    variants = [
        dict(fusion_type="hybrid_proposed"),
        dict(fusion_type="hybrid_proposed", use_cross_attention=False),
        dict(fusion_type="hybrid_proposed", use_gate=False),
        dict(fusion_type="hybrid_proposed", use_contradiction=False),
        dict(fusion_type="hybrid_proposed", use_social=False),
        dict(fusion_type="late_concat"),
        dict(fusion_type="late_concat", use_social=False),
        dict(fusion_type="text_only"),
        dict(fusion_type="image_only"),
        dict(fusion_type="social_only"),
    ]
    for v in variants:
        vcfg = VMTFConfig(**v, **common)
        fusion = MultimodalFusion(vcfg)
        out, gw = fusion(
            text_tokens=text_tokens if vcfg.uses_text else None,
            text_cls=text_cls if vcfg.uses_text else None,
            patch_tokens=patch_tokens if vcfg.uses_image else None,
            visual_cls=visual_cls if vcfg.uses_image else None,
            social_feat=es if vcfg.uses_social else None,
            attention_mask=mask,
            image_mask=image_mask if vcfg.uses_image else None,
        )
        assert out.shape == (B, vcfg.fusion_dim), f"{vcfg.experiment_name()} wrong shape"
        print(f"    - {vcfg.experiment_name():<55} OK (gate: {'yes' if gw else 'no'})")
    print("    [OK] All fusion variants passed.")

    # 6. Classifier Head
    print("\n[6] Testing ClassifierHead...")
    classifier = ClassifierHead(cfg.fusion_dim, cfg.classifier_hidden_dim, cfg.num_classes, cfg.classifier_dropout)
    assert classifier(fused).shape == (B, 2)
    print("    [OK] ClassifierHead passed.")

    print("\n" + "=" * 60)
    print("[OK] ALL STEP 2 MODULE CHECKS PASSED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    run_tests()
