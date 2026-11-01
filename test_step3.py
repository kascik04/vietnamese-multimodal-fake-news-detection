"""
Unit Testing Script for VMTF-Net Losses (FocalLoss, TextImageConsistencyLoss, VMTFCombinedLoss).
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


def test_losses() -> None:
    try:
        import torch
        import torch.nn.functional as F
        from vmtf_net.losses import FocalLoss, TextImageConsistencyLoss, VMTFCombinedLoss
    except ImportError as e:
        print(f"[!] PyTorch is not installed in local environment: {e}")
        return

    print("=" * 60)
    print("RUNNING VMTF-NET LOSSES CHECKS")
    print("=" * 60)

    B, D = 4, 256
    tmp = tempfile.mkdtemp()
    common = dict(checkpoint_dir=os.path.join(tmp, "c"), log_dir=os.path.join(tmp, "l"), device="cpu")

    # 1. Focal Loss
    print("\n[1] Testing FocalLoss...")
    logits = torch.tensor([[2.5, -1.0], [-0.5, 3.0], [1.2, 0.8], [-2.0, 1.5]])
    targets = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    focal_val = FocalLoss(gamma=2.0)(logits, targets)
    ce_val = F.cross_entropy(logits, targets)
    assert 0.0 < focal_val.item() < ce_val.item(), "Focal loss must be positive and below CE"
    assert torch.allclose(FocalLoss(gamma=0.0)(logits, targets), ce_val, atol=1e-6), "gamma=0 must equal CE"
    weighted = FocalLoss(gamma=0.0, weight=torch.tensor([1.0, 3.0]))(logits, targets)
    assert weighted.item() > 0.0
    print(f"    - Focal Loss: {focal_val.item():.4f} (CE {ce_val.item():.4f})")
    print("    [OK] FocalLoss passed.")

    # 2. Text-Image Consistency Loss
    print("\n[2] Testing TextImageConsistencyLoss...")
    consistency_fn = TextImageConsistencyLoss(margin=0.2)
    text_emb = torch.randn(B, D)
    real = torch.zeros(B, dtype=torch.long)
    fake = torch.ones(B, dtype=torch.long)
    assert consistency_fn(text_emb, text_emb.clone(), real).item() < 1e-4, "Aligned real -> ~0"
    assert consistency_fn(text_emb, -text_emb, real).item() > 1.9, "Opposite real -> ~2"
    assert consistency_fn(text_emb, -text_emb, fake).item() < 1e-6, "Opposite fake -> 0"
    assert abs(consistency_fn(text_emb, text_emb.clone(), fake).item() - 0.8) < 1e-4, "Aligned fake -> 1 - margin"

    # Missing-image samples are excluded from the average
    no_img = torch.tensor([1.0, 1.0, 0.0, 0.0])
    mixed_visual = torch.cat([text_emb[:2], -text_emb[2:]])
    assert consistency_fn(text_emb, mixed_visual, real, image_mask=no_img).item() < 1e-4
    assert consistency_fn(text_emb, mixed_visual, real, image_mask=torch.zeros(B)).item() == 0.0
    print("    [OK] TextImageConsistencyLoss passed.")

    # 3. Combined Loss
    print("\n[3] Testing VMTFCombinedLoss...")
    cfg = VMTFConfig(consistency_lambda=0.15, **common)
    combined = VMTFCombinedLoss(cfg)
    outputs = {"logits": logits, "text_proj": text_emb, "visual_proj": -text_emb}
    total, parts = combined(outputs, real)
    expected = F.cross_entropy(logits, real) + 0.15 * consistency_fn(text_emb, -text_emb, real)
    assert torch.allclose(total, expected, atol=1e-5)
    assert set(parts) == {"total_loss", "cls_loss", "consistency_loss"}
    print(f"    - Total {float(parts['total_loss']):.4f} | Cls {float(parts['cls_loss']):.4f} | "
          f"Consistency {float(parts['consistency_loss']):.4f}")

    # Unimodal outputs have no projections -> consistency term disabled
    total_uni, parts_uni = combined({"logits": logits, "text_proj": None, "visual_proj": None}, real)
    assert float(parts_uni["consistency_loss"]) == 0.0
    assert torch.allclose(total_uni, F.cross_entropy(logits, real))

    # Gradient flows into logits and projections
    lg = logits.clone().requires_grad_(True)
    tp = text_emb.clone().requires_grad_(True)
    total_g, _ = combined({"logits": lg, "text_proj": tp, "visual_proj": -text_emb}, real)
    total_g.backward()
    assert lg.grad is not None and tp.grad is not None
    print("    [OK] VMTFCombinedLoss passed.")

    print("\n" + "=" * 60)
    print("[OK] ALL LOSS CHECKS PASSED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    test_losses()
