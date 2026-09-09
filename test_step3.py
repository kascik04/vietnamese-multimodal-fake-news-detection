"""
Unit Testing Script for VMTF-Net Losses (FocalLoss, TextImageConsistencyLoss, VMTFCombinedLoss).
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


def test_losses() -> None:
    try:
        import torch
        import torch.nn.functional as F
        from vmtf_net.losses import FocalLoss, TextImageConsistencyLoss, VMTFCombinedLoss
    except ImportError as e:
        print(f"[!] PyTorch is not installed in local environment: {e}")
        print("[!] Ready to test seamlessly on Kaggle GPU / Google Colab!")
        return

    print("=" * 60)
    print("RUNNING VMTF-NET LOSSES CHECKS")
    print("=" * 60)

    B = 4
    D = 768

    # 1. Focal Loss Test
    print("\n[1] Testing FocalLoss...")
    focal_fn = FocalLoss(gamma=2.0)
    mock_logits = torch.tensor([[2.5, -1.0], [-0.5, 3.0], [1.2, 0.8], [-2.0, 1.5]])
    mock_targets = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    focal_val = focal_fn(mock_logits, mock_targets)
    print(f"    - Focal Loss Value: {focal_val.item():.4f}")
    assert focal_val.item() > 0.0, "Focal loss must be positive!"
    print("    [OK] FocalLoss passed.")

    # 2. Text-Image Consistency Loss Test
    print("\n[2] Testing TextImageConsistencyLoss...")
    consistency_fn = TextImageConsistencyLoss(margin=0.2)
    # Perfectly aligned text and image
    text_cls = torch.randn(B, D)
    visual_cls_aligned = text_cls.clone()  # Cosine sim = 1.0
    # Dissimilar image
    visual_cls_opposite = -text_cls.clone()  # Cosine sim = -1.0

    # For Real News (targets = 0): aligned should have ~0 loss, opposite should have ~2.0 loss
    real_targets = torch.zeros(B, dtype=torch.long)
    loss_aligned_real = consistency_fn(text_cls, visual_cls_aligned, real_targets)
    loss_opposite_real = consistency_fn(text_cls, visual_cls_opposite, real_targets)

    print(f"    - Aligned Real Loss:   {loss_aligned_real.item():.6f} (Expected ~0.0)")
    print(f"    - Mismatched Real Loss:{loss_opposite_real.item():.4f} (Expected ~2.0)")
    assert loss_aligned_real.item() < 1e-4, "Aligned vectors must yield near zero loss for real news!"
    assert loss_opposite_real.item() > 1.9, "Opposite vectors must yield high loss for real news!"
    print("    [OK] TextImageConsistencyLoss passed.")

    # 3. Combined Loss Test
    print("\n[3] Testing VMTFCombinedLoss...")
    cfg = VMTFConfig(consistency_lambda=0.15)
    combined_loss_fn = VMTFCombinedLoss(cfg)

    mock_outputs = {
        "logits": mock_logits,
        "text_cls": text_cls,
        "visual_cls": visual_cls_aligned,
    }
    total_loss, loss_dict = combined_loss_fn(mock_outputs, mock_targets)
    print(f"    - Total Loss:       {loss_dict['total_loss']:.4f}")
    print(f"    - Cls Loss:         {loss_dict['cls_loss']:.4f}")
    print(f"    - Consistency Loss: {loss_dict['consistency_loss']:.4f}")
    assert "total_loss" in loss_dict
    assert "cls_loss" in loss_dict
    assert "consistency_loss" in loss_dict
    print("    [OK] VMTFCombinedLoss passed.")

    print("\n" + "=" * 60)
    print("[OK] ALL LOSS CHECKS PASSED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    test_losses()
