"""
Verification and Mock Testing Script for VMTF-Net Step 1.
Generates synthetic Vietnamese fake news data (text, images, social features)
and validates configuration loading, dataset preprocessing, and batch collation.
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


def generate_mock_data(data_dir: str = "data") -> None:
    """Generate mock Vietnamese news dataset and sample images for testing."""
    import pandas as pd
    from PIL import Image, ImageDraw

    os.makedirs(os.path.join(data_dir, "images"), exist_ok=True)

    # 1. Create dummy images
    img_names = ["news_1.jpg", "news_2.jpg", "news_3.jpg", "news_4.jpg"]
    colors = [(220, 50, 50), (50, 180, 50), (50, 100, 220), (200, 200, 50)]

    for name, color in zip(img_names, colors):
        img_path = os.path.join(data_dir, "images", name)
        img = Image.new("RGB", (300, 300), color=color)
        draw = ImageDraw.Draw(img)
        draw.text((10, 10), name, fill=(255, 255, 255))
        img.save(img_path)

    # 2. Create sample Vietnamese multimodal records using standard schema columns
    mock_records = [
        {
            "sample_id": "post_001",
            "text": "Bộ Y Tế chính thức bác bỏ tin đồn thất thiệt lan truyền trên mạng xã hội sáng nay.",
            "image_1_local_path": "news_1.jpg",
            "like_count": 1250,
            "comment_count": 340,
            "share_count": 120,
            "angry_count": 15,
            "haha_count": 30,
            "label": 0,  # Real News
        },
        {
            "sample_id": "post_002",
            "text": "Cảnh báo khẩn cấp: Phương pháp chữa bách bệnh bằng lá cây chưa qua kiểm chứng khoa học!",
            "image_1_local_path": "news_2.jpg",
            "like_count": 89000,
            "comment_count": 15400,
            "share_count": 45000,
            "angry_count": 31150,
            "haha_count": 35600,
            "label": 1,  # Fake News
        },
        {
            "sample_id": "post_003",
            "text": "Chính phủ phê duyệt dự án phát triển công nghệ cao và trí tuệ nhân tạo quốc gia.",
            "image_1_local_path": "news_3.jpg",
            "like_count": 4500,
            "comment_count": 820,
            "share_count": 910,
            "angry_count": 10,
            "haha_count": 45,
            "label": 0,  # Real News
        },
        {
            "sample_id": "post_004",
            "text": "Sốc: Hình ảnh sinh vật lạ xuất hiện tại vùng biển miền Trung gây hoang mang dư luận.",
            "image_1_local_path": "news_4.jpg",
            "like_count": 52000,
            "comment_count": 8700,
            "share_count": 19500,
            "angry_count": 13000,
            "haha_count": 28600,
            "label": 1,  # Fake News
        },
    ]

    train_df = pd.DataFrame(mock_records)
    val_df = pd.DataFrame(mock_records[:2])

    train_csv = os.path.join(data_dir, "train.csv")
    val_csv = os.path.join(data_dir, "val.csv")

    train_df.to_csv(train_csv, index=False, encoding="utf-8")
    val_df.to_csv(val_csv, index=False, encoding="utf-8")
    print(f"[OK] Generated mock dataset at: {train_csv} and {val_csv}")


def test_config() -> None:
    """Test VMTFConfig initialization, serialization, and deserialization."""
    print("\n--- Testing VMTFConfig ---")
    cfg = VMTFConfig(
        batch_size=8,
        fusion_type="hybrid_proposed",
        consistency_lambda=0.15,
    )
    assert cfg.batch_size == 8
    assert cfg.social_input_dim == 3
    print(f"[OK] VMTFConfig initialized:\n{cfg}\n")

    # Test serialization
    cfg_json_path = "checkpoints/test_config.json"
    cfg.save(cfg_json_path)
    loaded_cfg = VMTFConfig.load(cfg_json_path)
    assert loaded_cfg.consistency_lambda == 0.15
    assert loaded_cfg.batch_size == 8
    print(f"[OK] Configuration successfully saved and reloaded from {cfg_json_path}")


def test_dataset_and_dataloader() -> None:
    """Test VMTFDataset and DataLoader tensor generation."""
    print("\n--- Testing VMTFDataset & DataLoader ---")
    try:
        import torch
        from transformers import AutoTokenizer
        from vmtf_net.dataset import create_dataloaders
    except ImportError as e:
        print(f"[!] Torch / Transformers not installed in current local environment: {e}")
        print("[!] Ready to test seamlessly on Kaggle GPU / Google Colab!")
        return

    cfg = VMTFConfig(
        batch_size=2,
        val_batch_size=2,
        max_seq_len=64,
        image_size=224,
        train_csv="data/train.csv",
        val_csv="data/val.csv",
        image_dir="data/images",
        num_workers=0,  # Safe for local test
        social_preset="extended",
    )

    print("Loading PhoBERT tokenizer & creating dataloaders...")
    train_loader, val_loader, _, normalizer = create_dataloaders(
        config=cfg,
        train_data=cfg.train_csv,
        val_data=cfg.val_csv,
    )

    batch = next(iter(train_loader))
    print(f"[OK] Successfully retrieved training batch:")
    print(f"    - input_ids shape:       {list(batch['input_ids'].shape)}  (Expected: [2, 64])")
    print(f"    - attention_mask shape:  {list(batch['attention_mask'].shape)}  (Expected: [2, 64])")
    print(f"    - pixel_values shape:    {list(batch['pixel_values'].shape)}  (Expected: [2, 3, 224, 224])")
    print(f"    - social_features shape: {list(batch['social_features'].shape)}  (Expected: [2, 8])")
    print(f"    - label shape:           {list(batch['label'].shape)}  (Expected: [2])")
    print(f"    - labels:                {batch['label'].tolist()}")

    assert batch["input_ids"].shape == (2, 64), "Text input_ids shape mismatch!"
    assert batch["pixel_values"].shape == (2, 3, 224, 224), "Pixel values shape mismatch!"
    assert batch["social_features"].shape == (2, 8), "Social features shape mismatch!"
    assert batch["label"].shape == (2,), "Labels shape mismatch!"
    print("\n[OK] ALL STEP 1 CHECKS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    test_config()
    try:
        import pandas
        from PIL import Image
        generate_mock_data()
        test_dataset_and_dataloader()
    except ImportError as e:
        print(f"\n[Note] Local Python lacks some libraries ({e}).")
        print("When pushed to Kaggle or run inside an environment with requirements.txt, all tests run end-to-end.")
