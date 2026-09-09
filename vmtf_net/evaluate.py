"""
Evaluation script for VMTF-Net (Vietnamese Multimodal Transformer Fusion Network).
Calculates:
- Classification Metrics: Accuracy, Precision, Recall, Macro-F1, AUC-ROC.
- Detailed Classification Report and Confusion Matrix.
- Interpretability Analysis: Modality reliability weights (alpha vs beta) across Real vs Fake news.
- Generates structured JSON report.
"""

import argparse
import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import VMTFConfig
from .dataset import VMTFCollate, VMTFDataset
from .models.vmtf_net import VMTFNet

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def evaluate_model(
    model: VMTFNet,
    test_loader: DataLoader,
    device: torch.device,
) -> Dict[str, Any]:
    """
    Run comprehensive evaluation on test dataset.
    
    Returns dictionary with all metrics and interpretability breakdowns.
    """
    model.eval()
    all_targets = []
    all_preds = []
    all_probs = []
    alphas = []
    betas = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating", leave=True):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            pixel_values = batch["pixel_values"].to(device)
            social_features = batch["social_features"].to(device)
            targets = batch["label"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                social_features=social_features,
            )

            probs = outputs["probs"]
            preds = torch.argmax(probs, dim=-1)

            all_targets.extend(targets.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())
            # Probability of Fake class (label 1) for AUC-ROC
            all_probs.extend(probs[:, 1].cpu().numpy().tolist())

            gate_w = outputs.get("gate_weights", {})
            if "alpha" in gate_w and "beta" in gate_w:
                alphas.extend(gate_w["alpha"].squeeze(-1).cpu().numpy().tolist())
                betas.extend(gate_w["beta"].squeeze(-1).cpu().numpy().tolist())

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    # 1. Standard Metrics
    acc = float(accuracy_score(y_true, y_pred))
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    f1_binary = float(f1_score(y_true, y_pred, average="binary", zero_division=0))
    precision = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    recall = float(recall_score(y_true, y_pred, average="macro", zero_division=0))

    # 2. AUC-ROC (checks if test set contains both classes)
    try:
        auc_roc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.0
    except ValueError:
        auc_roc = 0.0

    # 3. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred).tolist()

    # 4. Classification Report
    target_names = ["Real News (0)", "Fake News (1)"]
    report = classification_report(
        y_true, y_pred, target_names=target_names, output_dict=True, zero_division=0
    )

    # 5. Interpretability Analysis
    interpretability = {}
    if len(alphas) > 0:
        alphas_np = np.array(alphas)
        betas_np = np.array(betas)

        real_mask = y_true == 0
        fake_mask = y_true == 1

        interpretability = {
            "overall_avg_alpha_text": float(np.mean(alphas_np)),
            "overall_avg_beta_visual": float(np.mean(betas_np)),
            "real_news_avg_alpha_text": float(np.mean(alphas_np[real_mask])) if np.any(real_mask) else 0.0,
            "real_news_avg_beta_visual": float(np.mean(betas_np[real_mask])) if np.any(real_mask) else 0.0,
            "fake_news_avg_alpha_text": float(np.mean(alphas_np[fake_mask])) if np.any(fake_mask) else 0.0,
            "fake_news_avg_beta_visual": float(np.mean(betas_np[fake_mask])) if np.any(fake_mask) else 0.0,
        }

    results = {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_binary": f1_binary,
        "precision": precision,
        "recall": recall,
        "auc_roc": auc_roc,
        "confusion_matrix": cm,
        "classification_report": report,
        "interpretability": interpretability,
    }

    return results


def print_evaluation_summary(results: Dict[str, Any]) -> None:
    """Print clean formatted summary to console."""
    print("\n" + "=" * 65)
    print("        VMTF-NET EVALUATION RESULTS ON TEST SET")
    print("=" * 65)
    print(f"  Accuracy:       {results['accuracy'] * 100:.2f}%")
    print(f"  Macro F1-Score: {results['f1_macro'] * 100:.2f}%")
    print(f"  Binary F1 (Fake): {results['f1_binary'] * 100:.2f}%")
    print(f"  Precision:      {results['precision'] * 100:.2f}%")
    print(f"  Recall:         {results['recall'] * 100:.2f}%")
    print(f"  AUC-ROC:        {results['auc_roc'] * 100:.2f}%")
    print("-" * 65)
    print("  Confusion Matrix [[TN, FP], [FN, TP]]:")
    for row in results["confusion_matrix"]:
        print(f"    {row}")
    print("-" * 65)
    interp = results.get("interpretability", {})
    if interp:
        print("  Reliability-Aware Gating Interpretability:")
        print(f"    - Overall Text Weight (alpha):   {interp.get('overall_avg_alpha_text', 0):.3f}")
        print(f"    - Overall Visual Weight (beta):  {interp.get('overall_avg_beta_visual', 0):.3f}")
        print(f"    - On Real News (Text / Visual):  {interp.get('real_news_avg_alpha_text', 0):.3f} / {interp.get('real_news_avg_beta_visual', 0):.3f}")
        print(f"    - On Fake News (Text / Visual):  {interp.get('fake_news_avg_alpha_text', 0):.3f} / {interp.get('fake_news_avg_beta_visual', 0):.3f}")
    print("=" * 65 + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained VMTF-Net checkpoint")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_vmtf_model.pt", help="Path to checkpoint")
    parser.add_argument("--test_csv", type=str, default="data/test.csv", help="Path to test CSV")
    parser.add_argument("--image_dir", type=str, default="data/images", help="Root directory of images")
    parser.add_argument("--output_json", type=str, default="logs/evaluation_results.json", help="Path to output JSON")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint file not found: {args.checkpoint}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Loading checkpoint from {args.checkpoint} on {device}")

    # 1. Load Checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config_dict = checkpoint.get("config", {})
    cfg = VMTFConfig.from_dict(config_dict)
    cfg.test_csv = args.test_csv
    cfg.image_dir = args.image_dir

    # 2. Build Model & Load Weights
    model = VMTFNet(cfg).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    logger.info(f"Model successfully loaded. Best training F1 was: {checkpoint.get('val_f1', 0.0):.4f}")

    # 3. Build Test Dataset & DataLoader
    test_dataset = VMTFDataset(
        data=cfg.test_csv,
        config=cfg,
        is_train=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.val_batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        collate_fn=VMTFCollate(),
    )

    # 4. Run Evaluation
    results = evaluate_model(model, test_loader, device)
    print_evaluation_summary(results)

    # 5. Save Results JSON
    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    logger.info(f"Full evaluation results saved to: {args.output_json}")


if __name__ == "__main__":
    main()
