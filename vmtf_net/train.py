"""
Training script for VMTF-Net (Vietnamese Multimodal Transformer Fusion Network).
Optimized for Kaggle GPU (T4 x2 / P100), Colab, and Local Workstations.
Features:
- Automatic Mixed Precision (AMP), gradient accumulation, optional gradient checkpointing.
- Multi-GPU through DataParallel (Kaggle T4 x2) - checkpoints always store the unwrapped model.
- Differential learning rate grouping for transformer backbones vs fusion heads.
- Two checkpoints: `best_vmtf_model.pt` (weights + normalizer + threshold, small) and
  `last_checkpoint.pt` (optimizer/scheduler/scaler/RNG state) for `--resume` after a
  Kaggle/Colab session timeout. Both are written atomically.
- Per-epoch history CSV, run_info.json (versions, GPU, params, timings).
- Final evaluation: threshold tuned on validation, then applied once on the test split.
- CLI flags for every ablation (fusion type, cross-attention, gate, contradiction, social, lambda).
"""

import argparse
import json
import logging
import math
import os
import platform
import random
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from .config import FUSION_TYPES, VMTFConfig
    from .dataset import SocialFeatureNormalizer, compute_class_weights, create_dataloaders
    from .evaluate import (
        CHECKPOINT_FORMAT_VERSION,
        build_report,
        compute_metrics,
        find_best_threshold,
        autocast_context,
        json_default,
        move_batch,
        print_evaluation_summary,
        resolve_amp_dtype,
        run_inference,
        save_predictions,
        setup_cuda_precision,
    )
    from .losses import VMTFCombinedLoss
    from .models.vmtf_net import VMTFNet
except (ImportError, ValueError):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from vmtf_net.config import FUSION_TYPES, VMTFConfig
    from vmtf_net.dataset import SocialFeatureNormalizer, compute_class_weights, create_dataloaders
    from vmtf_net.evaluate import (
        CHECKPOINT_FORMAT_VERSION,
        build_report,
        compute_metrics,
        find_best_threshold,
        autocast_context,
        json_default,
        move_batch,
        print_evaluation_summary,
        resolve_amp_dtype,
        run_inference,
        save_predictions,
        setup_cuda_precision,
    )
    from vmtf_net.losses import VMTFCombinedLoss
    from vmtf_net.models.vmtf_net import VMTFNet

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    """Seed every random generator (DataLoader workers are seeded in dataset.py)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def atomic_save(obj: Any, path: str) -> None:
    """Write to a temp file then rename, so a killed session never leaves a corrupt checkpoint."""
    tmp_path = path + ".tmp"
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)


def environment_info() -> Dict[str, Any]:
    """Library versions and hardware, saved for reproducibility."""
    import transformers

    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count(),
        "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "kaggle": os.path.exists("/kaggle"),
        "colab": "COLAB_GPU" in os.environ or "COLAB_RELEASE_TAG" in os.environ,
    }
    return info


class VMTFTrainer:
    """
    Orchestrates training, validation, checkpointing, resuming and early stopping for VMTF-Net.
    """

    def __init__(
        self,
        config: VMTFConfig,
        model: VMTFNet,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        social_normalizer: Optional[SocialFeatureNormalizer] = None,
        class_weights: Optional[torch.Tensor] = None,
        save_last: bool = True,
        style_normalizer: Optional[SocialFeatureNormalizer] = None,
    ) -> None:
        self.config = config
        self.device = torch.device(config.device)
        self.base_model = model.to(self.device)
        self.model: torch.nn.Module = self.base_model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.social_normalizer = social_normalizer
        self.save_last = save_last
        self.style_normalizer = style_normalizer

        # Multi-GPU (Kaggle T4 x2). DataParallel needs no launcher, unlike DDP, inside notebooks.
        self.num_gpus = torch.cuda.device_count() if self.device.type == "cuda" else 0
        if config.multi_gpu and self.num_gpus > 1:
            logger.info(f"Using DataParallel on {self.num_gpus} GPUs (batch {config.batch_size} is split across GPUs).")
            self.model = torch.nn.DataParallel(self.base_model)

        # 1. Loss function
        if class_weights is not None:
            logger.info(f"Class weights: {class_weights.tolist()}")
            class_weights = class_weights.to(self.device)
        self.criterion = VMTFCombinedLoss(config, class_weights=class_weights).to(self.device)

        # 2. Optimizer with differential learning rates
        self.optimizer = torch.optim.AdamW(self.base_model.get_parameter_groups())

        # 3. Learning Rate Scheduler (Linear Warmup + Cosine Decay), counted in optimizer steps
        self.steps_per_epoch = max(1, math.ceil(len(train_loader) / config.grad_accum_steps))
        total_steps = self.steps_per_epoch * config.num_epochs
        warmup_steps = int(total_steps * config.warmup_ratio)
        try:
            from transformers import get_cosine_schedule_with_warmup

            self.scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps,
            )
        except ImportError:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=total_steps)

        # 4. Mixed Precision Scaler (fp16; T4/P100 have no bf16)
        # fp16 on T4 / P100 / V100 needs loss scaling; bf16 (A100, auto) has the fp32 range and does not
        self.use_amp = config.use_amp and self.device.type == "cuda"
        self.amp_dtype = resolve_amp_dtype(config.amp_dtype) if self.use_amp else torch.float32
        use_scaler = self.use_amp and self.amp_dtype == torch.float16
        try:
            self.scaler = torch.amp.GradScaler(self.device.type, enabled=use_scaler)
        except (AttributeError, TypeError):  # torch < 2.3
            self.scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)

        # Checkpointing / early-stopping state
        self.start_epoch = 0
        self.best_val_f1 = -1.0
        self.best_val_loss = float("inf")
        self.best_epoch = 0
        self.patience_counter = 0
        self.history: List[Dict[str, Any]] = []
        self.epoch_times: List[float] = []

        self.best_path = os.path.join(config.checkpoint_dir, "best_vmtf_model.pt")
        self.last_path = os.path.join(config.checkpoint_dir, "last_checkpoint.pt")
        self.history_path = os.path.join(config.checkpoint_dir, "history.csv")

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------
    def _save_best(self, epoch: int, val_metrics: Dict[str, Any], threshold: Optional[float] = None) -> None:
        atomic_save(
            {
                "format_version": CHECKPOINT_FORMAT_VERSION,
                "epoch": epoch,
                "model_state_dict": self.base_model.state_dict(),
                "config": self.config.to_dict(),
                "social_normalizer": self.social_normalizer.state_dict() if self.social_normalizer else None,
                "style_normalizer": self.style_normalizer.state_dict() if self.style_normalizer else None,
                "val_metrics": val_metrics,
                "best_threshold": threshold,
            },
            self.best_path,
        )

    def _save_last(self, epoch: int) -> None:
        atomic_save(
            {
                "format_version": CHECKPOINT_FORMAT_VERSION,
                "epoch": epoch,
                "model_state_dict": self.base_model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "scaler_state_dict": self.scaler.state_dict(),
                "config": self.config.to_dict(),
                "social_normalizer": self.social_normalizer.state_dict() if self.social_normalizer else None,
                "style_normalizer": self.style_normalizer.state_dict() if self.style_normalizer else None,
                "trainer_state": {
                    "best_val_f1": self.best_val_f1,
                    "best_val_loss": self.best_val_loss,
                    "best_epoch": self.best_epoch,
                    "patience_counter": self.patience_counter,
                    "history": self.history,
                    "epoch_times": self.epoch_times,
                },
                "rng_state": {
                    "python": random.getstate(),
                    "numpy": np.random.get_state(),
                    "torch": torch.get_rng_state(),
                    "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                },
            },
            self.last_path,
        )

    def resume(self, path: str) -> None:
        """Restore a `last_checkpoint.pt` and continue from the next epoch."""
        logger.info(f"Resuming from {path}")
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        self.base_model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.scaler.load_state_dict(ckpt["scaler_state_dict"])
        state = ckpt["trainer_state"]
        self.best_val_f1 = state["best_val_f1"]
        self.best_val_loss = state["best_val_loss"]
        self.best_epoch = state["best_epoch"]
        self.patience_counter = state["patience_counter"]
        self.history = state["history"]
        self.epoch_times = state.get("epoch_times", [])
        self.start_epoch = ckpt["epoch"]
        rng = ckpt.get("rng_state")
        if rng:
            random.setstate(rng["python"])
            np.random.set_state(rng["numpy"])
            torch.set_rng_state(rng["torch"])
            if rng.get("cuda") is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(rng["cuda"])
        logger.info(f"Resumed at epoch {self.start_epoch} (best val F1 so far {self.best_val_f1:.4f}).")

    # ------------------------------------------------------------------
    # Train / validate
    # ------------------------------------------------------------------
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Run one training epoch with gradient accumulation."""
        self.model.train()
        accum = self.config.grad_accum_steps
        sums = {"total_loss": 0.0, "cls_loss": 0.0, "consistency_loss": 0.0}
        sums_t = {k: torch.zeros((), device=self.device) for k in sums}
        all_preds, all_targets = [], []
        num_batches = len(self.train_loader)

        self.optimizer.zero_grad(set_to_none=True)
        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch+1}/{self.config.num_epochs} [Train]",
            leave=False,
            disable=self.config.disable_tqdm,
            mininterval=5,
        )

        for step, batch in enumerate(progress_bar):
            inputs = move_batch(batch, self.device)
            targets = batch["label"].to(self.device, non_blocking=True)

            with autocast_context(self.device, self.use_amp, self.amp_dtype):
                outputs = self.model(**inputs)
                loss, loss_dict = self.criterion(outputs, targets, inputs.get("image_mask"))

            self.scaler.scale(loss / accum).backward()

            if (step + 1) % accum == 0 or (step + 1) == num_batches:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.base_model.parameters(), self.config.grad_clip)
                scale_before = self.scaler.get_scale()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                # When AMP skips a step (inf/nan grads) the scale decreases: do not advance the LR schedule
                if not self.scaler.is_enabled() or self.scaler.get_scale() >= scale_before:
                    self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)

            for k in sums_t:
                sums_t[k] += loss_dict[k]
            all_preds.append(outputs["logits"].detach().argmax(dim=-1))
            all_targets.append(targets)

            if step % 20 == 0 and not self.config.disable_tqdm:
                progress_bar.set_postfix(loss=f"{loss_dict['total_loss'].item():.4f}")

        preds = torch.cat(all_preds).cpu().numpy()
        trues = torch.cat(all_targets).cpu().numpy()
        metrics = {k: float(v.item()) / num_batches for k, v in sums_t.items()}
        metrics["accuracy"] = float(accuracy_score(trues, preds))
        metrics["f1_macro"] = float(f1_score(trues, preds, average="macro", zero_division=0))
        return metrics

    def validate(self, data_loader: DataLoader, desc: str = "Validation") -> Dict[str, Any]:
        """Validation metrics at threshold 0.5 (the threshold is tuned only once, after training)."""
        out = run_inference(
            self.model, data_loader, self.device, self.use_amp, self.criterion, desc, self.config.disable_tqdm,
            amp_dtype=self.amp_dtype,
        )
        if out["y_true"] is None:
            raise ValueError("The validation split must contain valid 0/1 labels.")
        metrics = compute_metrics(out["y_true"], out["prob_fake"], 0.5)
        metrics["loss"] = out["loss"]
        if out["beta"] is not None:
            metrics["avg_beta"] = float(np.mean(out["beta"]))
        return metrics

    def fit(self) -> None:
        """Complete training workflow with checkpointing and early stopping."""
        logger.info(f"Starting training for {self.config.num_epochs} epochs on {self.device}")
        logger.info(f"Experiment: {self.config.experiment_name()} | AMP: {self.use_amp} ({str(self.amp_dtype).replace('torch.', '')})")

        for epoch in range(self.start_epoch, self.config.num_epochs):
            t0 = time.perf_counter()
            train_metrics = self.train_epoch(epoch)
            row: Dict[str, Any] = {"epoch": epoch + 1}
            row.update({f"train_{k}": v for k, v in train_metrics.items()})
            row["lr_backbone"] = self.optimizer.param_groups[0]["lr"]

            log_msg = (
                f"Epoch {epoch+1:02d}/{self.config.num_epochs:02d} | "
                f"Train Loss: {train_metrics['total_loss']:.4f} | Acc: {train_metrics['accuracy']:.4f} | "
                f"F1: {train_metrics['f1_macro']:.4f}"
            )

            if self.val_loader is not None:
                val_metrics = self.validate(self.val_loader, desc=f"Epoch {epoch+1} [Val]")
                row.update({
                    f"val_{k}": v for k, v in val_metrics.items()
                    if isinstance(v, (int, float)) or v is None
                })
                log_msg += (
                    f" || Val Loss: {val_metrics['loss']:.4f} | Acc: {val_metrics['accuracy']:.4f} | "
                    f"F1: {val_metrics['f1_macro']:.4f}"
                )
                if "avg_beta" in val_metrics:
                    log_msg += f" | Gate(T/V): {1 - val_metrics['avg_beta']:.2f}/{val_metrics['avg_beta']:.2f}"

                # Model selection on Macro-F1, ties broken by lower validation loss
                f1, vloss = val_metrics["f1_macro"], val_metrics["loss"]
                improved = f1 > self.best_val_f1 + 1e-6 or (
                    abs(f1 - self.best_val_f1) <= 1e-6 and vloss < self.best_val_loss
                )
                if improved:
                    self.best_val_f1, self.best_val_loss, self.best_epoch = f1, vloss, epoch + 1
                    self.patience_counter = 0
                    self._save_best(epoch + 1, val_metrics)
                    log_msg += f" => [SAVED BEST (F1: {f1:.4f})]"
                else:
                    self.patience_counter += 1
                    log_msg += f" (Patience: {self.patience_counter}/{self.config.early_stopping_patience})"
            else:
                # Without validation data the last epoch is kept
                self.best_epoch = epoch + 1
                self._save_best(epoch + 1, {})

            epoch_time = time.perf_counter() - t0
            self.epoch_times.append(epoch_time)
            row["epoch_seconds"] = epoch_time
            self.history.append(row)
            pd.DataFrame(self.history).to_csv(self.history_path, index=False)
            log_msg += f" | {epoch_time:.0f}s"
            logger.info(log_msg)

            if self.save_last:
                self._save_last(epoch + 1)

            if self.val_loader is not None and self.patience_counter >= self.config.early_stopping_patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs.")
                break

        logger.info(
            f"Training completed. Best epoch {self.best_epoch}, best Val F1: {self.best_val_f1:.4f}. "
            f"Checkpoint: {self.best_path}"
        )

    # ------------------------------------------------------------------
    # Final evaluation
    # ------------------------------------------------------------------
    def finalize(self, test_loader: Optional[DataLoader] = None, n_bootstrap: int = 1000) -> Dict[str, Any]:
        """
        Reload the best weights, tune the decision threshold on validation, store it in the best
        checkpoint, then evaluate ONCE on the test split. Writes JSON reports and prediction CSVs.
        """
        results: Dict[str, Any] = {"experiment": self.config.experiment_name(), "best_epoch": self.best_epoch}
        if not os.path.exists(self.best_path):
            logger.warning("No best checkpoint found: skipping final evaluation.")
            return results

        best = torch.load(self.best_path, map_location="cpu", weights_only=False)
        self.base_model.load_state_dict(best["model_state_dict"])

        threshold = 0.5
        if self.val_loader is not None:
            val_out = run_inference(
                self.model, self.val_loader, self.device, self.use_amp, self.criterion,
                "Validation (best)", self.config.disable_tqdm, amp_dtype=self.amp_dtype,
            )
            threshold = find_best_threshold(val_out["y_true"], val_out["prob_fake"])
            results["val"] = build_report(val_out, threshold, n_bootstrap=0, seed=self.config.seed)
            save_predictions(os.path.join(self.config.checkpoint_dir, "val_predictions.csv"), val_out, threshold)
            best["best_threshold"] = threshold
            atomic_save(best, self.best_path)
            logger.info(f"Decision threshold tuned on validation: {threshold:.2f}")
        results["threshold"] = threshold

        if test_loader is not None:
            test_out = run_inference(
                self.model, test_loader, self.device, self.use_amp, self.criterion, "Testing", self.config.disable_tqdm,
                amp_dtype=self.amp_dtype,
            )
            results["test"] = build_report(test_out, threshold, n_bootstrap=n_bootstrap, seed=self.config.seed)
            save_predictions(os.path.join(self.config.checkpoint_dir, "test_predictions.csv"), test_out, threshold)
            print_evaluation_summary(results["test"], title=f"TEST RESULTS - {self.config.experiment_name()}")

        with open(os.path.join(self.config.checkpoint_dir, "results.json"), "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, ensure_ascii=False, default=json_default)
        return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train VMTF-Net for Vietnamese Fake News Detection")
    # Data
    parser.add_argument("--train_csv", type=str, default="data/train.csv", help="Path to train CSV")
    parser.add_argument("--val_csv", type=str, default="data/val.csv", help="Path to val CSV")
    parser.add_argument("--test_csv", type=str, default="data/test.csv", help="Path to test CSV (evaluated once at the end)")
    parser.add_argument("--image_dir", type=str, default="data/images", help="Root directory of images")
    parser.add_argument("--cache_dir", type=str, default=None, help="Cache dir for pre-tokenized text")
    parser.add_argument("--social_preset", type=str, default="basic", choices=["basic", "extended"],
                        help="Social features preset ('basic': 3 features, 'extended': 8 features)")
    # Backbones (hub ids or local paths for offline Kaggle)
    parser.add_argument("--text_model_name", type=str, default="vinai/phobert-base")
    parser.add_argument("--visual_model_name", type=str, default="google/vit-base-patch16-224")
    parser.add_argument("--max_seq_len", type=int, default=256)
    parser.add_argument("--text_pooling", type=str, default="cls", choices=["cls", "mean"])
    parser.add_argument("--word_segmenter", type=str, default="auto", choices=["auto", "vncorenlp", "pyvi", "none"])
    parser.add_argument("--vncorenlp_dir", type=str, default=None, help="VnCoreNLP model directory (needs Java)")
    parser.add_argument("--freeze_text_layers", type=int, default=0)
    parser.add_argument("--freeze_visual_layers", type=int, default=0)
    # Fusion / ablation
    parser.add_argument("--fusion_type", type=str, default="hybrid_proposed", choices=FUSION_TYPES)
    parser.add_argument("--no_cross_attention", action="store_true", help="Ablation: remove cross-modal attention")
    parser.add_argument("--no_gate", action="store_true", help="Ablation: fixed alpha = beta = 0.5")
    parser.add_argument("--no_contradiction", action="store_true", help="Ablation: remove contradiction feature")
    parser.add_argument("--no_social", action="store_true", help="Ablation: remove social branch")
    parser.add_argument("--use_style", action="store_true",
                        help="Add the stylistic / clickbait feature branch (hybrid, late_concat, text_only)")
    # Loss
    parser.add_argument("--consistency_lambda", type=float, default=0.1, help="Consistency loss weight (0 disables)")
    parser.add_argument("--use_focal_loss", action="store_true")
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--class_weighting", type=str, default="none", choices=["none", "balanced"])
    # Optimization
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size (16 for Kaggle T4)")
    parser.add_argument("--val_batch_size", type=int, default=32)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Trade compute for VRAM")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=2e-5, help="Backbone learning rate")
    parser.add_argument("--head_lr", type=float, default=1e-4, help="Classifier/Fusion head learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience (epochs)")
    parser.add_argument("--seed", type=int, default=42)
    # Runtime
    parser.add_argument("--num_workers", type=int, default=None, help="DataLoader workers (auto if omitted)")
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--amp_dtype", type=str, default="auto", choices=["auto", "fp16", "bf16"],
                        help="Mixed precision type: auto = bf16 on A100 (Ampere+), fp16 on T4/P100/V100")
    parser.add_argument("--no_tf32", action="store_true", help="Disable TF32 matmuls on Ampere+ GPUs")
    parser.add_argument("--no_multi_gpu", action="store_true", help="Use a single GPU even if more are visible")
    parser.add_argument("--no_tqdm", action="store_true", help="Disable progress bars (Kaggle 'Save & Run All')")
    parser.add_argument("--checkpoint_dir", type=str, default=None,
                        help="Output dir (default: ./checkpoints/<experiment_name>)")
    parser.add_argument("--log_dir", type=str, default="./logs")
    parser.add_argument("--resume", type=str, default=None, help="Path to last_checkpoint.pt to resume from")
    parser.add_argument("--no_save_last", action="store_true",
                        help="Do not write last_checkpoint.pt (~3GB with optimizer state); disables --resume")
    parser.add_argument("--keep_last", action="store_true",
                        help="Keep last_checkpoint.pt after a successful run (deleted by default to save disk)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Allow writing into a checkpoint_dir that already holds a finished/started run")
    parser.add_argument("--n_bootstrap", type=int, default=1000, help="Bootstrap resamples for test CIs")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    args = parse_args()

    cfg_kwargs: Dict[str, Any] = dict(
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        test_csv=args.test_csv,
        image_dir=args.image_dir,
        cache_dir=args.cache_dir,
        social_preset=args.social_preset,
        text_model_name=args.text_model_name,
        visual_model_name=args.visual_model_name,
        max_seq_len=args.max_seq_len,
        text_pooling=args.text_pooling,
        word_segmenter=args.word_segmenter,
        vncorenlp_dir=args.vncorenlp_dir,
        freeze_text_layers=args.freeze_text_layers,
        freeze_visual_layers=args.freeze_visual_layers,
        fusion_type=args.fusion_type,
        use_cross_attention=not args.no_cross_attention,
        use_gate=not args.no_gate,
        use_contradiction=not args.no_contradiction,
        use_social=not args.no_social,
        use_style_features=args.use_style,
        consistency_lambda=args.consistency_lambda,
        use_focal_loss=args.use_focal_loss,
        focal_gamma=args.focal_gamma,
        class_weighting=args.class_weighting,
        batch_size=args.batch_size,
        val_batch_size=args.val_batch_size,
        grad_accum_steps=args.grad_accum_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        head_learning_rate=args.head_lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        early_stopping_patience=args.patience,
        seed=args.seed,
        multi_gpu=not args.no_multi_gpu,
        amp_dtype=args.amp_dtype,
        allow_tf32=not args.no_tf32,
        disable_tqdm=args.no_tqdm,
        checkpoint_dir=args.checkpoint_dir or "./checkpoints",
        log_dir=args.log_dir,
    )
    if args.num_workers is not None:
        cfg_kwargs["num_workers"] = args.num_workers
    if args.no_amp:
        cfg_kwargs["use_amp"] = False
    cfg = VMTFConfig(**cfg_kwargs)
    if args.checkpoint_dir is None:
        cfg.checkpoint_dir = os.path.join("./checkpoints", cfg.experiment_name())
        os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    # Never silently overwrite another experiment's results
    existing = [f for f in ("best_vmtf_model.pt", "last_checkpoint.pt", "results.json")
                if os.path.exists(os.path.join(cfg.checkpoint_dir, f))]
    if existing and not args.resume and not args.overwrite:
        raise SystemExit(
            f"'{cfg.checkpoint_dir}' already contains {existing}. Use another --checkpoint_dir, "
            f"--resume {os.path.join(cfg.checkpoint_dir, 'last_checkpoint.pt')} to continue, "
            f"or --overwrite to replace it."
        )
    cfg.save(os.path.join(cfg.checkpoint_dir, "config.json"))
    print(cfg)

    set_seed(cfg.seed)
    setup_cuda_precision(cfg.allow_tf32)
    run_start = time.perf_counter()

    # 1. Build DataLoaders (social normalizer fitted on train only)
    train_loader, val_loader, test_loader, social_normalizer = create_dataloaders(
        config=cfg,
        train_data=cfg.train_csv,
        val_data=cfg.val_csv if cfg.val_csv and os.path.exists(cfg.val_csv) else None,
        test_data=cfg.test_csv if cfg.test_csv and os.path.exists(cfg.test_csv) else None,
    )
    if val_loader is None:
        logger.warning("No validation split: no early stopping / threshold tuning; the last epoch is kept.")

    class_weights = None
    if cfg.class_weighting == "balanced":
        class_weights = compute_class_weights(train_loader.dataset.labels, cfg.num_classes)

    # 2. Build VMTF-Net
    model = VMTFNet(cfg)
    params = model.count_parameters()
    logger.info(f"Parameters: total {params['total']:,} | trainable {params['trainable']:,}")

    # 3. Train
    trainer = VMTFTrainer(
        config=cfg,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        social_normalizer=social_normalizer,
        class_weights=class_weights,
        style_normalizer=train_loader.dataset.style_normalizer,
        save_last=not args.no_save_last,
    )
    if args.resume:
        trainer.resume(args.resume)
    trainer.fit()

    # 4. Final evaluation (threshold from val, single pass on test)
    results = trainer.finalize(test_loader, n_bootstrap=args.n_bootstrap)

    run_info = {
        "experiment": cfg.experiment_name(),
        "environment": environment_info(),
        "parameters": params,
        "precision": {"amp": trainer.use_amp, "amp_dtype": str(trainer.amp_dtype).replace("torch.", ""),
                      "tf32": bool(torch.backends.cuda.matmul.allow_tf32) if torch.cuda.is_available() else False,
                      "num_workers": cfg.num_workers, "gpus_used": max(trainer.num_gpus, 0)},
        "train_samples": len(train_loader.dataset),
        "val_samples": len(val_loader.dataset) if val_loader is not None else 0,
        "test_samples": len(test_loader.dataset) if test_loader is not None else 0,
        "best_epoch": trainer.best_epoch,
        "avg_epoch_seconds": float(np.mean(trainer.epoch_times)) if trainer.epoch_times else None,
        "total_run_seconds": time.perf_counter() - run_start,
        "peak_gpu_memory_gb": (torch.cuda.max_memory_allocated() / 1024 ** 3) if torch.cuda.is_available() else None,
        "test_samples_per_second": results.get("test", {}).get("samples_per_second"),
    }
    with open(os.path.join(cfg.checkpoint_dir, "run_info.json"), "w", encoding="utf-8") as f:
        json.dump(run_info, f, indent=4, ensure_ascii=False, default=json_default)

    # The run is complete: the resume checkpoint (~3GB) is no longer needed
    if not args.keep_last and os.path.exists(trainer.last_path):
        os.remove(trainer.last_path)
        logger.info("Removed last_checkpoint.pt (run finished; use --keep_last to keep it).")
    logger.info(f"All artifacts saved in {cfg.checkpoint_dir}")


if __name__ == "__main__":
    main()
