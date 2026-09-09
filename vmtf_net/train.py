"""
Training script for VMTF-Net (Vietnamese Multimodal Transformer Fusion Network).
Optimized for Kaggle GPU (T4/P100), Colab, and Local Workstations.
Features:
- Automatic Mixed Precision (AMP) for 2x speedup and 50% VRAM reduction.
- Differential learning rate grouping for transformer backbones vs fusion heads.
- Checkpointing, early stopping, and validation logging (Loss, Accuracy, Macro F1).
- CLI argument parsing for ablation experiments.
"""

import argparse
import logging
import os
import random
import sys
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import VMTFConfig
from .dataset import create_dataloaders
from .losses import VMTFCombinedLoss
from .models.vmtf_net import VMTFNet

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    """Ensure full reproducibility across random generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class VMTFTrainer:
    """
    Orchestrates training, validation, checkpointing, and early stopping for VMTF-Net.
    """

    def __init__(
        self,
        config: VMTFConfig,
        model: VMTFNet,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ) -> None:
        self.config = config
        self.device = torch.device(config.device)
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader

        # 1. Loss function
        self.criterion = VMTFCombinedLoss(config).to(self.device)

        # 2. Optimizer with differential learning rates
        param_groups = self.model.get_parameter_groups()
        self.optimizer = torch.optim.AdamW(param_groups)

        # 3. Learning Rate Scheduler (Linear Warmup + Cosine Decay)
        total_steps = len(train_loader) * config.num_epochs
        warmup_steps = int(total_steps * config.warmup_ratio)
        try:
            from transformers import get_cosine_schedule_with_warmup
            self.scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps,
            )
        except ImportError:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=total_steps
            )

        # 4. Mixed Precision Scaler for Kaggle T4 GPU
        self.use_amp = config.use_amp and self.device.type == "cuda"
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        # Checkpointing state
        self.best_val_f1 = -1.0
        self.best_val_loss = float("inf")
        self.patience_counter = 0

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Run one training epoch."""
        self.model.train()
        total_loss = 0.0
        total_cls_loss = 0.0
        total_consistency_loss = 0.0
        all_preds = []
        all_targets = []

        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch+1}/{self.config.num_epochs} [Train]",
            leave=False,
        )

        for batch in progress_bar:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            pixel_values = batch["pixel_values"].to(self.device)
            social_features = batch["social_features"].to(self.device)
            targets = batch["label"].to(self.device)

            self.optimizer.zero_grad()

            # Mixed precision forward pass
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    social_features=social_features,
                )
                loss, loss_dict = self.criterion(outputs, targets)

            # Backward pass with gradient scaling
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.scheduler is not None:
                self.scheduler.step()

            # Record metrics
            total_loss += loss_dict["total_loss"]
            total_cls_loss += loss_dict["cls_loss"]
            total_consistency_loss += loss_dict["consistency_loss"]

            preds = torch.argmax(outputs["logits"], dim=-1).detach().cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(targets.detach().cpu().numpy())

            progress_bar.set_postfix(
                loss=f"{loss_dict['total_loss']:.4f}",
                cls=f"{loss_dict['cls_loss']:.4f}",
            )

        num_batches = len(self.train_loader)
        epoch_acc = accuracy_score(all_targets, all_preds)
        epoch_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)

        return {
            "loss": total_loss / num_batches,
            "cls_loss": total_cls_loss / num_batches,
            "consistency_loss": total_consistency_loss / num_batches,
            "accuracy": epoch_acc,
            "f1_macro": epoch_f1,
        }

    @torch.no_grad()
    def evaluate(self, data_loader: DataLoader, desc: str = "Validation") -> Dict[str, float]:
        """Run evaluation on validation or test set."""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []
        alphas = []
        betas = []

        for batch in tqdm(data_loader, desc=desc, leave=False):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            pixel_values = batch["pixel_values"].to(self.device)
            social_features = batch["social_features"].to(self.device)
            targets = batch["label"].to(self.device)

            with torch.cuda.amp.autocast(enabled=self.use_amp):
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    social_features=social_features,
                )
                loss, _ = self.criterion(outputs, targets)

            total_loss += loss.item()
            preds = torch.argmax(outputs["logits"], dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(targets.cpu().numpy())

            # Track reliability weights for interpretability
            gate_w = outputs.get("gate_weights", {})
            if "alpha" in gate_w and "beta" in gate_w:
                alphas.extend(gate_w["alpha"].squeeze(-1).cpu().numpy())
                betas.extend(gate_w["beta"].squeeze(-1).cpu().numpy())

        num_batches = len(data_loader)
        acc = accuracy_score(all_targets, all_preds)
        f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
        precision = precision_score(all_targets, all_preds, average="macro", zero_division=0)
        recall = recall_score(all_targets, all_preds, average="macro", zero_division=0)

        results = {
            "loss": total_loss / num_batches,
            "accuracy": acc,
            "f1_macro": f1,
            "precision": precision,
            "recall": recall,
        }

        if len(alphas) > 0:
            results["avg_alpha"] = float(np.mean(alphas))
            results["avg_beta"] = float(np.mean(betas))

        return results

    def fit(self) -> None:
        """Complete training workflow with checkpointing and early stopping."""
        logger.info(f"Starting training for {self.config.num_epochs} epochs on {self.device}")
        logger.info(f"Fusion strategy: {self.config.fusion_type} | AMP: {self.use_amp}")

        checkpoint_path = os.path.join(self.config.checkpoint_dir, "best_vmtf_model.pt")

        for epoch in range(self.config.num_epochs):
            train_metrics = self.train_epoch(epoch)
            
            log_msg = (
                f"Epoch {epoch+1:02d}/{self.config.num_epochs:02d} | "
                f"Train Loss: {train_metrics['loss']:.4f} | Acc: {train_metrics['accuracy']:.4f} | F1: {train_metrics['f1_macro']:.4f}"
            )

            if self.val_loader is not None:
                val_metrics = self.evaluate(self.val_loader, desc=f"Epoch {epoch+1} [Val]")
                log_msg += (
                    f" || Val Loss: {val_metrics['loss']:.4f} | Acc: {val_metrics['accuracy']:.4f} | F1: {val_metrics['f1_macro']:.4f}"
                )
                if "avg_alpha" in val_metrics:
                    log_msg += f" | Gate(T/V): {val_metrics['avg_alpha']:.2f}/{val_metrics['avg_beta']:.2f}"

                # Checkpointing on Macro F1
                if val_metrics["f1_macro"] > self.best_val_f1:
                    self.best_val_f1 = val_metrics["f1_macro"]
                    self.best_val_loss = val_metrics["loss"]
                    self.patience_counter = 0

                    torch.save(
                        {
                            "epoch": epoch + 1,
                            "model_state_dict": self.model.state_dict(),
                            "optimizer_state_dict": self.optimizer.state_dict(),
                            "val_f1": self.best_val_f1,
                            "config": self.config.to_dict(),
                        },
                        checkpoint_path,
                    )
                    log_msg += f" => [SAVED BEST CHECKPOINT (F1: {self.best_val_f1:.4f})]"
                else:
                    self.patience_counter += 1
                    log_msg += f" (Patience: {self.patience_counter}/{self.config.early_stopping_patience})"

            logger.info(log_msg)

            # Early stopping check
            if self.val_loader is not None and self.patience_counter >= self.config.early_stopping_patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs.")
                break

        logger.info(f"Training completed. Best Val F1: {self.best_val_f1:.4f}. Checkpoint saved at {checkpoint_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train VMTF-Net for Vietnamese Fake News Detection")
    parser.add_argument("--train_csv", type=str, default="data/train.csv", help="Path to train CSV")
    parser.add_argument("--val_csv", type=str, default="data/val.csv", help="Path to val CSV")
    parser.add_argument("--image_dir", type=str, default="data/images", help="Root directory of images")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size (16 for Kaggle T4)")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=2e-5, help="Backbone learning rate")
    parser.add_argument("--head_lr", type=float, default=1e-4, help="Classifier/Fusion head learning rate")
    parser.add_argument("--fusion_type", type=str, default="hybrid_proposed", choices=["hybrid_proposed", "late_concat"])
    parser.add_argument("--consistency_lambda", type=float, default=0.1, help="Consistency loss weight")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints", help="Directory for checkpoints")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = VMTFConfig(
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        image_dir=args.image_dir,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        head_learning_rate=args.head_lr,
        fusion_type=args.fusion_type,
        consistency_lambda=args.consistency_lambda,
        checkpoint_dir=args.checkpoint_dir,
    )

    set_seed(cfg.seed)

    # 1. Build DataLoaders
    train_loader, val_loader, _, _ = create_dataloaders(
        config=cfg,
        train_data=cfg.train_csv,
        val_data=cfg.val_csv if os.path.exists(cfg.val_csv) else None,
    )

    # 2. Build VMTF-Net
    model = VMTFNet(cfg)

    # 3. Train
    trainer = VMTFTrainer(
        config=cfg,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
    )
    trainer.fit()


if __name__ == "__main__":
    main()
