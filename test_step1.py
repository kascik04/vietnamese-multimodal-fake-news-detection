"""
Verification Script for VMTF-Net Step 1 (Config, Dataset, DataLoader) - runs fully OFFLINE.
Generates synthetic Vietnamese fake news data (text, images, social features) in a temp dir and
validates configuration round-trip, label filtering, missing-image mask, social normalizer
serialization, dynamic padding collation and leakage-free normalization.
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


class FakeTokenizer:
    """Minimal offline stand-in for the PhoBERT tokenizer (<s>=0, <pad>=1, </s>=2)."""

    pad_token_id = 1

    def __call__(self, texts, max_length=256, truncation=True, padding=False, **kwargs):
        ids = []
        for t in texts:
            body = [3 + (sum(map(ord, w)) % 90) for w in t.split()][: max_length - 2]
            ids.append([0] + body + [2])
        return {"input_ids": ids}


class FakeImageProcessor:
    image_mean = [0.5, 0.5, 0.5]
    image_std = [0.5, 0.5, 0.5]


def generate_mock_data(data_dir: str) -> None:
    """Generate mock Vietnamese news dataset and sample images for testing."""
    import pandas as pd
    from PIL import Image, ImageDraw

    os.makedirs(os.path.join(data_dir, "images"), exist_ok=True)
    colors = [(220, 50, 50), (50, 180, 50), (50, 100, 220), (200, 200, 50), (90, 90, 90), (10, 200, 200)]
    for i, color in enumerate(colors, start=1):
        img = Image.new("RGB", (300, 260), color=color)
        ImageDraw.Draw(img).text((10, 10), f"news_{i}", fill=(255, 255, 255))
        img.save(os.path.join(data_dir, "images", f"news_{i}.jpg"))

    def rec(i, text, like, comment, share, angry, haha, label, image=None):
        return {
            "sample_id": f"post_{i:03d}",
            "text": text,
            "image_1_local_path": image if image is not None else f"news_{i}.jpg",
            "like_count": like, "comment_count": comment, "share_count": share,
            "angry_count": angry, "haha_count": haha, "label": label,
        }

    train = [
        rec(1, "Bộ Y Tế chính thức bác bỏ tin đồn thất thiệt lan truyền trên mạng xã hội sáng nay.", 1250, 340, 120, 15, 30, 0),
        rec(2, "Cảnh báo khẩn cấp: Phương pháp chữa bách bệnh bằng lá cây chưa qua kiểm chứng khoa học!", 89000, 15400, 45000, 31150, 35600, 1),
        rec(3, "Chính phủ phê duyệt dự án phát triển công nghệ cao và trí tuệ nhân tạo quốc gia.", 4500, 820, 910, 10, 45, 0),
        rec(4, "Sốc: Hình ảnh sinh vật lạ xuất hiện tại vùng biển miền Trung gây hoang mang dư luận.", 52000, 8700, 19500, 13000, 28600, 1, image="missing_file.jpg"),
        rec(9, "Dòng có nhãn không hợp lệ phải bị loại bỏ.", 1, 1, 1, 0, 0, 7),
    ]
    # Validation posts are DIFFERENT from training posts (no leakage)
    val = [
        rec(5, "Ngân hàng Nhà nước điều chỉnh lãi suất điều hành từ tháng tới.", 3000, 400, 300, 5, 12, 0),
        rec(6, "Chia sẻ gấp: uống nước chanh nóng diệt virus trong 5 phút!", 70000, 9000, 30000, 20000, 15000, 1),
    ]
    pd.DataFrame(train).to_csv(os.path.join(data_dir, "train.csv"), index=False, encoding="utf-8")
    pd.DataFrame(val).to_csv(os.path.join(data_dir, "val.csv"), index=False, encoding="utf-8")
    print(f"[OK] Generated mock dataset in: {data_dir}")


def test_config(tmp_dir: str) -> None:
    """Test VMTFConfig initialization, validation, serialization and runtime-field handling."""
    print("\n--- Testing VMTFConfig ---")
    cfg = VMTFConfig(batch_size=8, fusion_type="hybrid_proposed", consistency_lambda=0.15,
                     checkpoint_dir=os.path.join(tmp_dir, "ckpt"), log_dir=os.path.join(tmp_dir, "logs"))
    assert cfg.batch_size == 8 and cfg.social_input_dim == 3
    print(f"[OK] VMTFConfig initialized:\n{cfg}\n")

    path = os.path.join(tmp_dir, "ckpt", "test_config.json")
    cfg.save(path)
    loaded = VMTFConfig.load(path)
    assert loaded.consistency_lambda == 0.15 and loaded.batch_size == 8

    # Hardware fields must be re-detected, not copied from another machine
    d = cfg.to_dict()
    d["num_workers"] = 99
    assert VMTFConfig.from_dict(d).num_workers != 99
    assert VMTFConfig.from_dict(d, restore_runtime=True).num_workers == 99

    ext = VMTFConfig(social_preset="extended", checkpoint_dir=cfg.checkpoint_dir, log_dir=cfg.log_dir)
    assert ext.social_input_dim == 8
    try:
        VMTFConfig(fusion_type="bogus", checkpoint_dir=cfg.checkpoint_dir, log_dir=cfg.log_dir)
        raise AssertionError("Invalid fusion_type must raise")
    except ValueError:
        pass
    assert "no_gate" in VMTFConfig(use_gate=False, checkpoint_dir=cfg.checkpoint_dir, log_dir=cfg.log_dir).experiment_name()
    print("[OK] Config save/load, runtime-field reset, presets and validation passed.")


def test_dataset_and_dataloader(tmp_dir: str) -> None:
    """Test VMTFDataset, social normalizer serialization and DataLoader tensor generation."""
    print("\n--- Testing VMTFDataset & DataLoader ---")
    try:
        import numpy as np
        import torch
        from vmtf_net.dataset import SocialFeatureNormalizer, create_dataloaders, parse_label
    except ImportError as e:
        print(f"[!] Torch not installed in current environment: {e}")
        return

    data_dir = os.path.join(tmp_dir, "data")
    generate_mock_data(data_dir)

    cfg = VMTFConfig(
        batch_size=2,
        val_batch_size=2,
        max_seq_len=16,
        image_size=224,
        image_dir=os.path.join(data_dir, "images"),
        num_workers=0,
        social_preset="extended",
        word_segmenter="none",
        checkpoint_dir=os.path.join(tmp_dir, "ckpt"),
        log_dir=os.path.join(tmp_dir, "logs"),
    )

    assert parse_label("2") is None and parse_label("fake") == 1 and parse_label(0) == 0

    train_loader, val_loader, _, normalizer = create_dataloaders(
        config=cfg,
        train_data=os.path.join(data_dir, "train.csv"),
        val_data=os.path.join(data_dir, "val.csv"),
        tokenizer=FakeTokenizer(),
        image_processor=FakeImageProcessor(),
    )

    train_ds = train_loader.dataset
    assert len(train_ds) == 4, "Row with invalid label 7 must be dropped"
    assert train_ds.has_image_flags.tolist() == [True, True, True, False], "Missing image must be flagged"

    # Val uses the TRAIN normalizer (no refit on val)
    assert val_loader.dataset.social_normalizer is normalizer

    # Normalizer survives a checkpoint round-trip
    restored = SocialFeatureNormalizer.from_state_dict(normalizer.state_dict())
    assert np.allclose(restored.transform(train_ds.df), normalizer.transform(train_ds.df), atol=1e-6)

    batch = next(iter(val_loader))
    L = batch["input_ids"].shape[1]
    print("[OK] Successfully retrieved validation batch:")
    print(f"    - input_ids shape:       {list(batch['input_ids'].shape)}  (dynamic padding, <= {cfg.max_seq_len})")
    print(f"    - pixel_values shape:    {list(batch['pixel_values'].shape)}  (Expected: [2, 3, 224, 224])")
    print(f"    - has_image:             {batch['has_image'].tolist()}")
    print(f"    - social_features shape: {list(batch['social_features'].shape)}  (Expected: [2, 8])")
    print(f"    - labels:                {batch['label'].tolist()}")

    assert L <= cfg.max_seq_len
    assert batch["attention_mask"].shape == batch["input_ids"].shape
    assert int(batch["attention_mask"].sum(dim=1).max()) == L, "Longest sequence must be unpadded"
    assert batch["pixel_values"].shape == (2, 3, 224, 224)
    assert batch["social_features"].shape == (2, 8)
    assert batch["label"].shape == (2,)

    # Full pass over train (includes the missing-image sample)
    for b in train_loader:
        assert torch.isfinite(b["pixel_values"]).all()

    # Style / clickbait branch: features computed on raw text, normalizer fitted on train only
    from vmtf_net.style_features import STYLE_FEATURE_NAMES

    cfg.use_style_features = True
    s_train, s_val, _, _ = create_dataloaders(
        config=cfg,
        train_data=os.path.join(data_dir, "train.csv"),
        val_data=os.path.join(data_dir, "val.csv"),
        tokenizer=FakeTokenizer(),
        image_processor=FakeImageProcessor(),
    )
    assert s_val.dataset.style_normalizer is s_train.dataset.style_normalizer
    s_batch = next(iter(s_val))
    assert s_batch["style_features"].shape == (2, len(STYLE_FEATURE_NAMES))
    assert torch.isfinite(s_batch["style_features"]).all()
    print(f"    - style_features shape:  {list(s_batch['style_features'].shape)}")

    # RAM image cache must give exactly the same eval pixels as decoding from disk
    cfg.use_style_features = False
    cfg.cache_images = True
    c_train, c_val, _, _ = create_dataloaders(
        config=cfg,
        train_data=os.path.join(data_dir, "train.csv"),
        val_data=os.path.join(data_dir, "val.csv"),
        tokenizer=FakeTokenizer(),
        image_processor=FakeImageProcessor(),
    )
    assert c_val.dataset.image_cache is not None and c_val.dataset.image_cache.dtype == np.uint8
    assert torch.equal(next(iter(c_val))["pixel_values"], batch["pixel_values"]), "cache changed the pixels"
    assert c_train.dataset.has_image_flags.tolist() == [True, True, True, False]
    for b in c_train:
        assert b["pixel_values"].shape[1:] == (3, 224, 224)
    cfg.cache_images = False

    # head_tail truncation keeps <s>, the beginning, the end and </s>
    from vmtf_net.dataset import head_tail_truncate

    long_ids = [0] + list(range(100, 200)) + [2]
    cut = head_tail_truncate(long_ids, 16)
    assert len(cut) == 16 and cut[0] == 0 and cut[-1] == 2 and cut[1] == 100 and cut[-2] == 199
    cfg.text_truncation = "head_tail"
    h_train, _, _, _ = create_dataloaders(
        config=cfg, train_data=os.path.join(data_dir, "train.csv"),
        tokenizer=FakeTokenizer(), image_processor=FakeImageProcessor(),
    )
    assert max(len(ids) for ids in h_train.dataset.input_ids) <= cfg.max_seq_len
    print("    - image cache + head_tail truncation: OK")
    cfg.text_truncation = "head"

    # Several images per post + OCR text appended to the post text
    import pandas as pd

    multi_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    multi_df["image_2_local_path"] = ["news_2.jpg", "", "missing.jpg", "news_1.jpg", ""]
    multi_df["image_3_local_path"] = ["news_3.jpg", "", "", "", ""]
    multi_df["ocr_text"] = ["THÔNG BÁO KHẨN của UBND", "", "", "", ""]
    cfg.max_images = 3
    cfg.use_ocr_text = True
    cfg.max_seq_len = 64
    for cached in (False, True):
        cfg.cache_images = cached
        m_train, _, _, _ = create_dataloaders(
            config=cfg, train_data=multi_df, tokenizer=FakeTokenizer(), image_processor=FakeImageProcessor(),
        )
        ds = m_train.dataset
        assert [len(p) for p in ds.image_lists] == [3, 1, 1, 1], ds.image_lists   # row 4 (bad label) dropped
        item0, item3 = ds[0], ds[3]
        assert item0["pixel_values"].shape == (3, 3, 224, 224)
        assert item0["image_slot_mask"].tolist() == [1.0, 1.0, 1.0] and item0["has_image"].item() == 1.0
        # post 4: image_1 is missing on disk, image_2 exists -> it fills the first slot
        assert item3["image_slot_mask"].tolist() == [1.0, 0.0, 0.0]
        assert torch.all(item3["pixel_values"][1:] == 0)
        mb = next(iter(m_train))
        assert mb["pixel_values"].shape[1:] == (3, 3, 224, 224) and mb["image_slot_mask"].shape[1] == 3
    assert "Chữ trong ảnh" in ds.model_texts[0] and ds.model_texts[1] == ds.raw_texts[1]
    assert len(ds.input_ids[0]) > len(FakeTokenizer()([ds.raw_texts[0]])["input_ids"][0]), "OCR text not tokenized"
    assert "Chữ trong ảnh" not in ds.raw_texts[0], "style features must see the post text only"
    cfg.max_images, cfg.use_ocr_text, cfg.cache_images = 1, False, False
    print("    - multi-image (max_images=3) + OCR text: OK")
    print("\n[OK] ALL STEP 1 CHECKS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp:
        test_config(tmp)
        try:
            import pandas  # noqa: F401
            from PIL import Image  # noqa: F401
        except ImportError as e:
            print(f"\n[Note] Local Python lacks some libraries ({e}).")
        else:
            test_dataset_and_dataloader(tmp)
