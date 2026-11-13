"""
End-to-end OFFLINE smoke test for VMTF-Net Step 4 (full model, trainer, checkpoints, evaluation).
Builds TINY randomly-initialized RoBERTa (PhoBERT architecture) and ViT backbones locally,
so nothing is downloaded and it runs in seconds on CPU. Validates:
1. VMTFNet forward/backward for every fusion type and ablation.
2. Parameter groups (no weight decay on biases / norms).
3. VMTFTrainer.fit (AMP-safe loop, grad accumulation), history CSV, best/last checkpoints.
4. Resume from last_checkpoint.pt.
5. finalize(): threshold tuned on val, results.json, prediction CSVs.
6. evaluate.load_checkpoint() restores identical predictions (incl. social normalizer).
"""

import json
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

HIDDEN, IMG, SEQ = 32, 32, 12


def build_tiny_backbones(root: str):
    from transformers import RobertaConfig, RobertaModel, ViTConfig, ViTModel

    text_dir, vis_dir = os.path.join(root, "tiny_phobert"), os.path.join(root, "tiny_vit")
    RobertaModel(RobertaConfig(
        vocab_size=100, hidden_size=HIDDEN, num_hidden_layers=2, num_attention_heads=4,
        intermediate_size=64, max_position_embeddings=SEQ + 4, pad_token_id=1,
    ), add_pooling_layer=False).save_pretrained(text_dir)
    ViTModel(ViTConfig(
        hidden_size=HIDDEN, num_hidden_layers=2, num_attention_heads=4, intermediate_size=64,
        image_size=IMG, patch_size=16,
    ), add_pooling_layer=False).save_pretrained(vis_dir)
    return text_dir, vis_dir


def make_config(tmp: str, text_dir: str, vis_dir: str, **overrides) -> VMTFConfig:
    params = dict(
        text_model_name=text_dir, visual_model_name=vis_dir,
        text_dim=HIDDEN, visual_dim=HIDDEN, fusion_dim=HIDDEN, cross_attn_heads=4,
        social_hidden_dim=16, classifier_hidden_dim=16, consistency_proj_dim=8,
        image_size=IMG, max_seq_len=SEQ, batch_size=4, val_batch_size=4, num_epochs=2,
        device="cpu", num_workers=0, disable_tqdm=True, word_segmenter="none",
        checkpoint_dir=os.path.join(tmp, "ckpt"), log_dir=os.path.join(tmp, "logs"),
    )
    params.update(overrides)
    return VMTFConfig(**params)


def make_batches(n_batches: int, batch_size: int, seed: int, n_images: int = 1):
    """Synthetic collated batches (fake label correlates with a token & social signal)."""
    import torch
    from vmtf_net.dataset import VMTFCollate
    from vmtf_net.style_features import STYLE_FEATURE_NAMES

    g = torch.Generator().manual_seed(seed)
    collate = VMTFCollate(pad_token_id=1)
    batches = []
    for b in range(n_batches):
        items = []
        for i in range(batch_size):
            label = (b * batch_size + i) % 2
            length = int(torch.randint(4, SEQ, (1,), generator=g))
            ids = torch.randint(3, 100, (length,), generator=g)
            ids[0], ids[-1] = 0, 2
            ids[1] = 50 + label
            has_image = float(i != 0)  # first sample of every batch has no image
            if n_images > 1:
                # sample i has min(i, n_images) images: 0, 1, 2, ...
                slot_mask = (torch.arange(n_images) < i).float()
                pixels = torch.randn(n_images, 3, IMG, IMG, generator=g) * slot_mask.view(-1, 1, 1, 1)
            else:
                slot_mask = None
                pixels = torch.randn(3, IMG, IMG, generator=g) * has_image
            items.append({
                "id": f"s{b}_{i}",
                "input_ids": ids,
                "pixel_values": pixels,
                "has_image": torch.tensor(has_image),
                **({"image_slot_mask": slot_mask} if slot_mask is not None else {}),
                "social_features": torch.randn(3, generator=g) + label,
                "style_features": torch.randn(len(STYLE_FEATURE_NAMES), generator=g) + label,
                "label": torch.tensor(label),
            })
        batches.append(collate(items))
    return batches


def run_tests() -> None:
    try:
        import numpy as np
        import pandas as pd
        import torch
        import transformers  # noqa: F401
    except ImportError as e:
        print(f"[!] torch / transformers not installed: {e}")
        return

    from vmtf_net.dataset import SocialFeatureNormalizer
    from vmtf_net.evaluate import load_checkpoint, load_style_normalizer, move_batch, run_inference
    from vmtf_net.losses import VMTFCombinedLoss
    from vmtf_net.models.vmtf_net import VMTFNet
    from vmtf_net.train import VMTFTrainer, set_seed

    print("=" * 60)
    print("RUNNING VMTF-NET STEP 4 END-TO-END CHECKS (tiny offline backbones)")
    print("=" * 60)

    tmp = tempfile.mkdtemp()
    text_dir, vis_dir = build_tiny_backbones(tmp)
    set_seed(0)
    batch = make_batches(1, 4, seed=1)[0]

    # 1. Forward / backward for every experiment
    print("\n[1] Forward/backward for all fusion types and ablations...")
    variants = [
        dict(), dict(use_cross_attention=False), dict(use_gate=False), dict(use_contradiction=False),
        dict(use_social=False), dict(fusion_type="late_concat"), dict(fusion_type="text_only"),
        dict(fusion_type="image_only"), dict(fusion_type="social_only"), dict(text_pooling="mean"),
        dict(gradient_checkpointing=True), dict(use_style_features=True),
        dict(fusion_type="text_only", use_style_features=True), dict(consistency_lambda=0.0),
    ]
    for v in variants:
        cfg = make_config(tmp, text_dir, vis_dir, **v)
        model = VMTFNet(cfg)
        model.train()
        out = model(**move_batch(batch, torch.device("cpu")))
        loss, _ = VMTFCombinedLoss(cfg)(out, batch["label"], batch.get("has_image"))
        loss.backward()
        assert out["logits"].shape == (4, 2) and torch.isfinite(loss)
        assert (model.text_encoder is not None) == cfg.uses_text
        assert (model.visual_encoder is not None) == cfg.uses_image
        assert (model.style_encoder is not None) == cfg.uses_style
        assert (out["cosine_sim"] is not None) == (cfg.uses_text and cfg.uses_image and cfg.consistency_lambda > 0)
        if out["gate_weights"]:
            assert out["gate_weights"]["beta"][0].item() == 0.0, "No-image sample must have beta = 0"
        print(f"    - {cfg.experiment_name():<55} loss {loss.item():.4f}")
    # Several images per post: empty slots are never encoded and never change the output
    cfg_multi = make_config(tmp, text_dir, vis_dir, max_images=3)
    multi_batch = make_batches(1, 4, seed=5, n_images=3)[0]
    assert multi_batch["pixel_values"].shape == (4, 3, 3, IMG, IMG)
    model_multi = VMTFNet(cfg_multi).eval()
    with torch.no_grad():
        inputs = move_batch(multi_batch, torch.device("cpu"))
        out_a = model_multi(**inputs)
        garbage = dict(inputs)
        garbage["pixel_values"] = inputs["pixel_values"].clone()
        empty = inputs["image_slot_mask"] == 0
        garbage["pixel_values"][empty] = torch.randn(int(empty.sum()), 3, IMG, IMG)
        out_b = model_multi(**garbage)
    assert torch.allclose(out_a["logits"], out_b["logits"], atol=1e-5), "pixels of empty image slots leaked"
    assert out_a["gate_weights"]["beta"][0].item() == 0.0, "post without any image must have beta = 0"
    assert "img3" in cfg_multi.experiment_name()
    print(f"    - {cfg_multi.experiment_name():<55} multi-image masking OK")
    print("    [OK] All variants run forward + backward.")

    # 2. Parameter groups
    print("\n[2] Parameter groups...")
    cfg = make_config(tmp, text_dir, vis_dir)
    model = VMTFNet(cfg)
    groups = model.get_parameter_groups()
    n_grouped = sum(len(g["params"]) for g in groups)
    assert n_grouped == sum(1 for p in model.parameters() if p.requires_grad), "Every parameter must be grouped once"
    for g in groups:
        if g["weight_decay"] > 0:
            assert all(p.ndim >= 2 for p in g["params"]), "1-D params (bias/norm) must not be decayed"
    print(f"    - {len(groups)} groups, {n_grouped} tensors, params: {model.count_parameters()}")
    print("    [OK] Parameter groups passed.")

    # 3. Trainer fit + checkpoints
    print("\n[3] VMTFTrainer.fit (2 epochs, grad accumulation 2)...")
    cfg = make_config(tmp, text_dir, vis_dir, grad_accum_steps=2, early_stopping_patience=5, use_style_features=True)
    train_batches = make_batches(4, 4, seed=2)
    val_batches = make_batches(2, 4, seed=3)
    test_batches = make_batches(2, 4, seed=4)
    normalizer = SocialFeatureNormalizer(["share_count", "like_count", "comment_count"]).fit(
        pd.DataFrame({"share_count": [1, 10, 100], "like_count": [5, 50, 500], "comment_count": [0, 3, 30]})
    )
    from vmtf_net.style_features import STYLE_FEATURE_NAMES, extract_style_dataframe

    style_normalizer = SocialFeatureNormalizer(STYLE_FEATURE_NAMES).fit(
        extract_style_dataframe(["SỐC!!! chia sẻ gấp", "Thông báo lịch nghỉ lễ.", "KHẨN CẤP ⚠️"])
    )
    trainer = VMTFTrainer(cfg, VMTFNet(cfg), train_batches, val_batches, social_normalizer=normalizer,
                          style_normalizer=style_normalizer)
    trainer.fit()
    for name in ("best_vmtf_model.pt", "last_checkpoint.pt", "history.csv"):
        assert os.path.exists(os.path.join(cfg.checkpoint_dir, name)), f"{name} missing"
    history = pd.read_csv(os.path.join(cfg.checkpoint_dir, "history.csv"))
    assert len(history) == 2 and "val_f1_macro" in history.columns
    print("    [OK] Training loop, history and checkpoints passed.")

    # 4. Resume
    print("\n[4] Resume from last_checkpoint.pt...")
    cfg_resume = make_config(tmp, text_dir, vis_dir, num_epochs=3, use_style_features=True,
                             checkpoint_dir=os.path.join(tmp, "ckpt_resume"))
    resumed = VMTFTrainer(cfg_resume, VMTFNet(cfg_resume), train_batches, val_batches, social_normalizer=normalizer,
                          style_normalizer=style_normalizer)
    resumed.resume(os.path.join(cfg.checkpoint_dir, "last_checkpoint.pt"))
    assert resumed.start_epoch == 2 and len(resumed.history) == 2
    resumed.fit()
    assert len(resumed.history) == 3, "Resumed run must continue at epoch 3"
    print("    [OK] Resume passed.")

    # 5. Finalize: threshold on val, single test pass
    print("\n[5] finalize() on test split...")
    results = trainer.finalize(test_batches, n_bootstrap=50)
    assert 0.0 < results["threshold"] < 1.0
    assert "metrics" in results["test"] and "pr_auc" in results["test"]["metrics"]
    for name in ("results.json", "test_predictions.csv", "val_predictions.csv"):
        assert os.path.exists(os.path.join(cfg.checkpoint_dir, name)), f"{name} missing"
    with open(os.path.join(cfg.checkpoint_dir, "results.json"), encoding="utf-8") as f:
        json.load(f)
    preds = pd.read_csv(os.path.join(cfg.checkpoint_dir, "test_predictions.csv"))
    assert {"id", "prob_fake", "pred", "label", "beta", "has_image"} <= set(preds.columns)
    print("    [OK] finalize passed.")

    # 6. Checkpoint reload gives identical predictions
    print("\n[6] evaluate.load_checkpoint round-trip...")
    model2, cfg2, norm2, ckpt = load_checkpoint(
        os.path.join(cfg.checkpoint_dir, "best_vmtf_model.pt"), torch.device("cpu")
    )
    assert norm2 is not None and np.allclose(norm2.scaler.mean_, normalizer.scaler.mean_)
    style2 = load_style_normalizer(ckpt)
    assert cfg2.uses_style and style2 is not None
    assert np.allclose(style2.scaler.mean_, style_normalizer.scaler.mean_)
    assert ckpt["best_threshold"] == results["threshold"]
    out_a = run_inference(trainer.base_model, test_batches, torch.device("cpu"), disable_tqdm=True)
    out_b = run_inference(model2, test_batches, torch.device("cpu"), disable_tqdm=True)
    assert np.allclose(out_a["prob_fake"], out_b["prob_fake"], atol=1e-5), "Reloaded model differs!"
    print("    [OK] Checkpoint round-trip passed.")

    # 7. Single-post prediction API (same preprocessing path as training)
    print("\n[7] VMTFPredictor...")
    from PIL import Image

    from vmtf_net.predict import VMTFPredictor

    class _Tok:
        pad_token_id = 1

        def __call__(self, texts, max_length=SEQ, truncation=True, padding=False, **kw):
            return {"input_ids": [[0] + [3 + (sum(map(ord, w)) % 90) for w in t.split()][: max_length - 2] + [2]
                                  for t in texts]}

    class _Proc:
        image_mean = [0.5, 0.5, 0.5]
        image_std = [0.5, 0.5, 0.5]

    img_path = os.path.join(tmp, "post.jpg")
    Image.new("RGB", (80, 60), (200, 40, 40)).save(img_path)
    predictor = VMTFPredictor(model2, cfg2, norm2, style2, threshold=ckpt["best_threshold"],
                              device=torch.device("cpu"), tokenizer=_Tok(), image_processor=_Proc())
    one = predictor.predict("SỐC!!! Chia sẻ gấp tin này", images=[img_path], like_count=1200, share_count=95)
    assert 0.0 <= one["prob_fake"] <= 1.0 and one["label"] in (0, 1) and one["has_image"] is True
    assert abs(one["alpha_text"] + one["beta_image"] - 1.0) < 1e-5 and one["style_signals"]
    text_only_post = predictor.predict("Thông báo lịch nghỉ lễ của trường.")
    assert text_only_post["has_image"] is False and text_only_post["beta_image"] == 0.0
    many = predictor.predict_batch([{"text": "bài một", "id": "a"}, {"text": "bài hai", "image": img_path, "id": "b"}])
    assert [m["id"] for m in many] == ["a", "b"] and many[1]["has_image"]
    again = predictor.predict("SỐC!!! Chia sẻ gấp tin này", images=[img_path], like_count=1200, share_count=95)
    assert abs(again["prob_fake"] - one["prob_fake"]) < 1e-6, "prediction must be deterministic"

    from vmtf_net.retrieval import PostChecker

    report = predictor.predict_and_check("Hôm nay trời đẹp.", PostChecker(None))
    assert report["prediction"]["prob_fake"] == report["model_prob_fake"] and "hint" in report
    print("    [OK] VMTFPredictor passed.")

    print("\n" + "=" * 60)
    print("[OK] ALL STEP 4 END-TO-END CHECKS PASSED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    run_tests()
