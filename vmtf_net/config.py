"""
Configuration module for VMTF-Net (Vietnamese Multimodal Transformer Fusion Network).
Provides dataclass-based settings with auto-detection for multi-environment execution
(Kaggle GPU, Google Colab, Local Workstations).
"""

import json
import os
import re
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
    - Kaggle / Colab: 2 workers (both expose only 2-4 vCPUs).
    - Windows: 0 (avoids multiprocessing spawn overhead/issues on Windows).
    """
    if os.name == "nt":
        return 0
    if os.path.exists("/kaggle") or "COLAB_GPU" in os.environ or "COLAB_RELEASE_TAG" in os.environ:
        return 2
    return min(4, os.cpu_count() or 1)


_GENERIC_PATH_PARTS = {"", "default", "main", "transformers", "pytorch", "model", "models", "input", "kaggle"}


def _short_model_name(name: str) -> str:
    """
    Readable short name of a hub id or local path, e.g.
    'vinai/phobert-base' -> 'phobert-base',
    '/kaggle/input/phobert-base/transformers/default/1' -> 'phobert-base'.
    """
    parts = [p for p in re.split(r"[\\/]+", name.strip()) if p]
    for part in reversed(parts):
        if part.lower() not in _GENERIC_PATH_PARTS and not part.isdigit():
            return re.sub(r"[^a-z0-9.\-]+", "-", part.lower()).strip("-") or "model"
    return "model"


BASIC_SOCIAL_COLS: List[str] = [
    "share_count",
    "like_count",
    "comment_count",
]

EXTENDED_SOCIAL_COLS: List[str] = [
    "like_count",
    "comment_count",
    "share_count",
    "angry_count",
    "haha_count",
    "like_share_ratio",
    "comment_share_ratio",
    "engagement_ratio",
]

FUSION_TYPES: List[str] = ["hybrid_proposed", "late_concat", "text_only", "image_only", "social_only"]

# Hardware-dependent fields: never restored from a checkpoint trained on another machine.
RUNTIME_FIELDS: List[str] = ["device", "use_amp", "num_workers", "pin_memory", "multi_gpu"]


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
    """HuggingFace backbone or local path: 'vinai/phobert-base', 'vinai/phobert-base-v2', 'uitnlp/visobert'"""

    text_dim: int = 768
    """Hidden dimension of text transformer backbone [B, L, 768]"""

    max_seq_len: int = 256
    """Maximum token sequence length (PhoBERT supports at most 256)"""

    freeze_text_layers: int = 0
    """Number of bottom transformer layers to freeze (0 to fine-tune all)"""

    text_pooling: str = "cls"
    """Sentence pooling: 'cls' (hidden state of <s>) or 'mean' (masked mean over tokens).
    The HF pooler is NOT used: PhoBERT/ViSoBERT checkpoints ship without pretrained pooler weights."""

    word_segmenter: str = "auto"
    """Vietnamese word segmentation: 'auto' (VnCoreNLP if available else pyvi, only for PhoBERT),
    'vncorenlp', 'pyvi', or 'none'. Syllable-level models (ViSoBERT) must use 'none'."""

    vncorenlp_dir: Optional[str] = None
    """Directory holding VnCoreNLP models (py_vncorenlp). Requires Java."""

    # ==========================================
    # 2. VISUAL ENCODER (Vision Transformer - ViT)
    # ==========================================
    visual_model_name: str = "google/vit-base-patch16-224"
    """HuggingFace ViT backbone or local path"""

    visual_dim: int = 768
    """Hidden dimension of visual transformer backbone [B, P, 768]"""

    image_size: int = 224
    """Input image resolution (224x224 for ViT patch16)"""

    freeze_visual_layers: int = 0
    """Number of bottom visual transformer layers to freeze (0 to fine-tune all)"""

    # ==========================================
    # 3. SOCIAL METRICS ENCODER (MLP)
    # ==========================================
    social_preset: str = "basic"
    """Social features preset: 'basic' (3 features) or 'extended' (8 features from dataset schema)"""

    social_feature_cols: List[str] = field(default_factory=lambda: list(BASIC_SOCIAL_COLS))
    """List of numerical engagement columns: [share_count, like_count, comment_count] or extended set"""

    social_input_dim: int = 3
    """Number of numerical input features (derived from social_feature_cols)"""

    social_hidden_dim: int = 128
    """Output representation dimension for social branch E_S in R^128"""

    social_dropout: float = 0.2
    """Dropout probability for social MLP"""

    # ==========================================
    # 4. FUSION MECHANISM & ABLATION STUDY
    # ==========================================
    fusion_type: str = "hybrid_proposed"
    """
    - 'hybrid_proposed': Cross-Attention (Contradiction) + Reliability-Aware Gated Fusion.
    - 'late_concat': Baseline simple concatenation [E_T; E_V; E_S].
    - 'text_only' / 'image_only' / 'social_only': unimodal baselines.
    """

    use_cross_attention: bool = True
    """Ablation (hybrid only): bidirectional cross-modal attention. If False, contradiction is computed on CLS."""

    use_gate: bool = True
    """Ablation (hybrid only): learned reliability gate. If False, alpha = beta = 0.5."""

    use_contradiction: bool = True
    """Ablation (hybrid only): contradiction feature [|T-V|; T*V]."""

    use_social: bool = True
    """Ablation (hybrid / late_concat): social engagement branch."""

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

    consistency_proj_dim: int = 256
    """Dimension of the shared text-image space in which the consistency loss is computed"""

    consistency_margin: float = 0.2
    """Cosine margin for fake samples in the consistency loss"""

    use_focal_loss: bool = False
    """Whether to use Focal Loss instead of CrossEntropy for imbalanced datasets"""

    focal_gamma: float = 2.0
    """Focusing parameter gamma for Focal Loss"""

    class_weighting: str = "none"
    """'none' or 'balanced' (inverse class frequency computed on the training split)"""

    # ==========================================
    # 7. OPTIMIZATION & TRAINING
    # ==========================================
    batch_size: int = 16
    """Training batch size per optimizer micro-step (16 fits a 16GB Kaggle T4 with AMP)"""

    val_batch_size: int = 32
    """Validation and testing batch size"""

    grad_accum_steps: int = 1
    """Gradient accumulation steps (effective batch = batch_size * grad_accum_steps)"""

    gradient_checkpointing: bool = False
    """Enable gradient checkpointing in both backbones (saves VRAM, ~30% slower)"""

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
    """Epochs without validation Macro-F1 improvement (ties broken by val loss) before early stop"""

    # ==========================================
    # 8. HARDWARE & MULTI-ENVIRONMENT CONFIG
    # ==========================================
    device: str = field(default_factory=_detect_device)
    """Target compute device ('cuda' or 'cpu')"""

    use_amp: bool = True
    """Enable Automatic Mixed Precision (fp16) on GPU"""

    num_workers: int = field(default_factory=_detect_default_num_workers)
    """Number of DataLoader worker processes (2 on Kaggle/Colab, 0 on Windows)"""

    pin_memory: bool = True
    """Pin memory in DataLoader for faster host-to-device GPU transfers"""

    multi_gpu: bool = True
    """Use all visible GPUs via DataParallel (e.g. Kaggle T4 x2)"""

    disable_tqdm: bool = False
    """Disable progress bars (recommended for Kaggle 'Save & Run All' logs)"""

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

    cache_dir: Optional[str] = None
    """Optional directory to cache pre-tokenized text (e.g. /kaggle/working/cache)"""

    checkpoint_dir: str = "./checkpoints"
    """Directory to store model checkpoints (.pt), history and results"""

    log_dir: str = "./logs"
    """Directory to store training logs and metrics"""

    def __post_init__(self) -> None:
        """Validate and adjust parameters post-initialization."""
        if self.social_preset == "extended" and self.social_feature_cols == BASIC_SOCIAL_COLS:
            self.social_feature_cols = list(EXTENDED_SOCIAL_COLS)
        elif self.social_preset == "basic" and self.social_feature_cols == EXTENDED_SOCIAL_COLS:
            self.social_feature_cols = list(BASIC_SOCIAL_COLS)
        self.social_input_dim = len(self.social_feature_cols)

        if self.fusion_type not in FUSION_TYPES:
            raise ValueError(f"Unknown fusion_type '{self.fusion_type}'. Choose one of {FUSION_TYPES}.")
        if self.text_pooling not in ("cls", "mean"):
            raise ValueError("text_pooling must be 'cls' or 'mean'.")
        if self.word_segmenter not in ("auto", "vncorenlp", "pyvi", "none"):
            raise ValueError("word_segmenter must be 'auto', 'vncorenlp', 'pyvi' or 'none'.")
        if self.class_weighting not in ("none", "balanced"):
            raise ValueError("class_weighting must be 'none' or 'balanced'.")
        self.grad_accum_steps = max(1, int(self.grad_accum_steps))

        # If running on CPU, disable AMP and pin_memory automatically
        if self.device == "cpu":
            self.use_amp = False
            self.pin_memory = False

        # Ensure directories exist if specified
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Modality helpers (used by model / dataset to skip unused branches)
    # ------------------------------------------------------------------
    @property
    def uses_text(self) -> bool:
        return self.fusion_type in ("hybrid_proposed", "late_concat", "text_only")

    @property
    def uses_image(self) -> bool:
        return self.fusion_type in ("hybrid_proposed", "late_concat", "image_only")

    @property
    def uses_social(self) -> bool:
        if self.fusion_type == "social_only":
            return True
        return self.fusion_type in ("hybrid_proposed", "late_concat") and self.use_social

    def experiment_name(self) -> str:
        """
        Tag describing the experiment (used as default output directory, logs and results).
        Includes everything that changes the model or the objective, so two different
        experiments never share (and overwrite) the same directory.
        """
        tag = self.fusion_type
        if self.fusion_type == "hybrid_proposed":
            off = [n for n, v in [("xattn", self.use_cross_attention), ("gate", self.use_gate),
                                  ("contra", self.use_contradiction), ("social", self.use_social)] if not v]
            if off:
                tag += "-no_" + "-no_".join(off)
        elif self.fusion_type == "late_concat" and not self.use_social:
            tag += "-no_social"

        parts = [tag]
        if self.uses_text:
            parts.append(_short_model_name(self.text_model_name))
            if self.text_pooling != "cls":
                parts.append(f"pool-{self.text_pooling}")
        if self.uses_image:
            parts.append(_short_model_name(self.visual_model_name))
        if self.uses_social:
            parts.append(f"social-{self.social_preset}")
        if self.use_focal_loss:
            parts.append(f"focal{self.focal_gamma}")
        if self.class_weighting != "none":
            parts.append(f"cw-{self.class_weighting}")
        if self.uses_text and self.uses_image:
            parts.append(f"lambda{self.consistency_lambda}")
        parts.append(f"seed{self.seed}")
        return "_".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any], restore_runtime: bool = False) -> "VMTFConfig":
        """
        Instantiate configuration from dictionary.
        Hardware-dependent fields (device, AMP, workers, ...) are re-detected on the current
        machine unless restore_runtime=True.
        """
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in config_dict.items() if k in valid_keys}
        if not restore_runtime:
            for k in RUNTIME_FIELDS:
                filtered.pop(k, None)
        return cls(**filtered)

    def save(self, filepath: str) -> None:
        """Save configuration as formatted JSON."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=4, ensure_ascii=False)

    @classmethod
    def load(cls, filepath: str, restore_runtime: bool = False) -> "VMTFConfig":
        """Load configuration from JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data, restore_runtime=restore_runtime)

    def __repr__(self) -> str:
        """Pretty string representation of key settings."""
        lines = [
            "VMTFConfig (Vietnamese Multimodal Transformer Fusion Network)",
            f"  - Device:              {self.device} (AMP: {self.use_amp}, Multi-GPU: {self.multi_gpu})",
            f"  - Text Backbone:       {self.text_model_name} (Max Seq: {self.max_seq_len}, Pool: {self.text_pooling}, Seg: {self.word_segmenter})",
            f"  - Visual Backbone:     {self.visual_model_name} (Image: {self.image_size}x{self.image_size})",
            f"  - Social Features ({self.social_input_dim}): {self.social_feature_cols}",
            f"  - Experiment:          {self.experiment_name()}",
            f"  - Batch Size:          {self.batch_size} x accum {self.grad_accum_steps} (Val: {self.val_batch_size})",
            f"  - Learning Rates:      Backbone: {self.learning_rate}, Head: {self.head_learning_rate}",
            f"  - Loss Config:         Lambda Consistency: {self.consistency_lambda}, Focal: {self.use_focal_loss}, Class weights: {self.class_weighting}",
            f"  - Num Workers:         {self.num_workers}",
        ]
        return "\n".join(lines)


if __name__ == "__main__":
    cfg = VMTFConfig()
    print(cfg)
