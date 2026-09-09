"""
Configuration module for VMTF-Net (Vietnamese Multimodal Transformer Fusion Network).
Provides dataclass-based settings with auto-detection for multi-environment execution 
(Kaggle GPU, Google Colab, Local Workstations).
"""

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


def _detect_device() -> str:
    """Safely detect available compute device."""
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _detect_default_num_workers() -> int:
    """
    Auto-detect optimal DataLoader workers depending on OS and platform.
    - Kaggle / Linux: 2 workers (balanced for I/O and CPU memory).
    - Windows: 0 (avoids multiprocessing spawn overhead/issues on Windows).
    """
    if os.name == "nt":
        return 0
    # Kaggle environment check
    if os.path.exists("/kaggle"):
        return 2
    return min(4, os.cpu_count() or 1)


@dataclass
class VMTFConfig:
    """
    Comprehensive configuration for VMTF-Net architecture, dataset, 
    loss function, and training pipeline.
    """

    # ==========================================
    # 1. TEXT ENCODER (PhoBERT / ViSoBERT)
    # ==========================================
    text_model_name: str = "vinai/phobert-base"
    """HuggingFace backbone: 'vinai/phobert-base', 'UIT-NLP/visobert', or 'vinai/phobert-base-v2'"""
    
    text_dim: int = 768
    """Hidden dimension of text transformer backbone [B, L, 768]"""
    
    max_seq_len: int = 256
    """Maximum token sequence length for text tokenizer"""
    
    freeze_text_layers: int = 0
    """Number of bottom transformer layers to freeze (0 to fine-tune all)"""

    # ==========================================
    # 2. VISUAL ENCODER (Vision Transformer - ViT)
    # ==========================================
    visual_model_name: str = "google/vit-base-patch16-224"
    """HuggingFace ViT backbone: 'google/vit-base-patch16-224'"""
    
    visual_dim: int = 768
    """Hidden dimension of visual transformer backbone [B, P, 768]"""
    
    image_size: int = 224
    """Input image resolution (224x224 for ViT patch16)"""
    
    freeze_visual_layers: int = 0
    """Number of bottom visual transformer layers to freeze (0 to fine-tune all)"""

    # ==========================================
    # 3. SOCIAL METRICS ENCODER (MLP)
    # ==========================================
    social_feature_cols: List[str] = field(
        default_factory=lambda: [
            "share_count",
            "like_count",
            "comment_count",
        ]
    )
    """List of numerical engagement columns: [share_count, like_count, comment_count] across Facebook/Threads/X"""
    
    social_input_dim: int = 3
    """Number of numerical input features (default 3: share_count, like_count, comment_count)"""

    
    social_hidden_dim: int = 128
    """Output representation dimension for social branch E_S in R^128"""
    
    social_dropout: float = 0.2
    """Dropout probability for social MLP"""

    # ==========================================
    # 4. FUSION MECHANISM & ABLATION STUDY
    # ==========================================
    fusion_type: str = "hybrid_proposed"
    """
    Ablation study options:
    - 'hybrid_proposed': Cross-Attention (Contradiction) + Reliability-Aware Gated Fusion.
    - 'late_concat': Baseline simple concatenation [E_T; E_V; E_S].
    """
    
    fusion_dim: int = 768
    """Intermediate projected dimension for cross-modal attention"""
    
    cross_attn_heads: int = 8
    """Number of attention heads in multi-head cross-modal attention"""
    
    cross_attn_dropout: float = 0.1
    """Dropout probability in cross-modal attention"""
    
    gate_dropout: float = 0.1
    """Dropout probability for reliability gating network"""

    # ==========================================
    # 5. CLASSIFIER HEAD
    # ==========================================
    num_classes: int = 2
    """Binary classification: 0 (Real News), 1 (Fake News)"""
    
    classifier_hidden_dim: int = 256
    """Hidden dimension of classification MLP"""
    
    classifier_dropout: float = 0.3
    """Dropout rate before final linear projection"""

    # ==========================================
    # 6. LOSS FUNCTIONS
    # ==========================================
    consistency_lambda: float = 0.1
    """Weight for text-image semantic consistency loss: Total_Loss = L_cls + lambda * L_consistency"""
    
    use_focal_loss: bool = False
    """Whether to use Focal Loss instead of CrossEntropy for imbalanced datasets"""
    
    focal_gamma: float = 2.0
    """Focusing parameter gamma for Focal Loss"""

    # ==========================================
    # 7. OPTIMIZATION & TRAINING
    # ==========================================
    batch_size: int = 16
    """Training batch size (16 is recommended for 16GB VRAM on Kaggle T4)"""
    
    val_batch_size: int = 32
    """Validation and testing batch size"""
    
    learning_rate: float = 2e-5
    """Learning rate for fine-tuning transformer backbones"""
    
    head_learning_rate: float = 1e-4
    """Higher learning rate for fusion layers, social MLP, and classifier head"""
    
    weight_decay: float = 0.01
    """AdamW weight decay"""
    
    num_epochs: int = 10
    """Maximum number of training epochs"""
    
    warmup_ratio: float = 0.1
    """Portion of total training steps for linear learning rate warmup"""
    
    grad_clip: float = 1.0
    """Gradient norm clipping threshold"""
    
    early_stopping_patience: int = 3
    """Number of epochs without validation loss improvement before early stop"""

    # ==========================================
    # 8. HARDWARE & MULTI-ENVIRONMENT CONFIG
    # ==========================================
    device: str = field(default_factory=_detect_device)
    """Target compute device ('cuda' or 'cpu')"""
    
    use_amp: bool = True
    """Enable Automatic Mixed Precision (torch.cuda.amp) on Kaggle T4 / GPU"""
    
    num_workers: int = field(default_factory=_detect_default_num_workers)
    """Number of DataLoader worker processes (2 on Kaggle/Linux, 0 on Windows)"""
    
    pin_memory: bool = True
    """Pin memory in DataLoader for faster host-to-device GPU transfers"""
    
    seed: int = 42
    """Random seed for full reproducibility"""

    # ==========================================
    # 9. PATHS & LOGGING
    # ==========================================
    train_csv: str = "data/train.csv"
    """Path to training CSV file"""
    
    val_csv: str = "data/val.csv"
    """Path to validation CSV file"""
    
    test_csv: str = "data/test.csv"
    """Path to test CSV file"""
    
    image_dir: str = "data/images"
    """Root directory containing social media images"""
    
    checkpoint_dir: str = "./checkpoints"
    """Directory to store model checkpoints (.pt)"""
    
    log_dir: str = "./logs"
    """Directory to store training logs and metrics"""

    def __post_init__(self) -> None:
        """Validate and adjust parameters post-initialization."""
        self.social_input_dim = len(self.social_feature_cols)
        
        # If running on CPU, disable AMP and pin_memory automatically
        if self.device == "cpu":
            self.use_amp = False
            self.pin_memory = False

        # Ensure directories exist if specified
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "VMTFConfig":
        """Instantiate configuration from dictionary."""
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in config_dict.items() if k in valid_keys}
        return cls(**filtered)

    def save(self, filepath: str) -> None:
        """Save configuration as formatted JSON."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=4, ensure_ascii=False)

    @classmethod
    def load(cls, filepath: str) -> "VMTFConfig":
        """Load configuration from JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def __repr__(self) -> str:
        """Pretty string representation of key settings."""
        lines = [
            "VMTFConfig (Vietnamese Multimodal Transformer Fusion Network)",
            f"  - Device:              {self.device} (AMP: {self.use_amp})",
            f"  - Text Backbone:       {self.text_model_name} (Max Seq: {self.max_seq_len})",
            f"  - Visual Backbone:     {self.visual_model_name} (Image: {self.image_size}x{self.image_size})",
            f"  - Social Features ({self.social_input_dim}): {self.social_feature_cols}",
            f"  - Fusion Strategy:     {self.fusion_type}",
            f"  - Batch Size:          {self.batch_size} (Val: {self.val_batch_size})",
            f"  - Learning Rates:      Backbone: {self.learning_rate}, Head: {self.head_learning_rate}",
            f"  - Loss Config:         Lambda Consistency: {self.consistency_lambda}, Focal: {self.use_focal_loss}",
            f"  - Num Workers:         {self.num_workers}",
        ]
        return "\n".join(lines)


if __name__ == "__main__":
    cfg = VMTFConfig()
    print(cfg)
