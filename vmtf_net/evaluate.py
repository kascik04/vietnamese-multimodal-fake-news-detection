"""
Evaluation script for VMTF-Net (Vietnamese Multimodal Transformer Fusion Network).
Calculates:
- Classification Metrics: Accuracy, Macro-F1, Fake-class F1/Precision/Recall, AUC-ROC, PR-AUC,
  with bootstrap 95% confidence intervals.
- Decision threshold tuned on the validation split (never on test).
- Interpretability: gate (alpha/beta) distributions and text-image cosine per class,
  plus performance on samples with vs. without images.
- Per-sample predictions CSV (input for scripts/significance_test.py) and JSON report.
"""

import argparse
import json
import logging
import os
import sys
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from .config import VMTFConfig
    from .dataset import SocialFeatureNormalizer, create_dataloaders, create_eval_loader
    from .models.vmtf_net import VMTFNet
except (ImportError, ValueError):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from vmtf_net.config import VMTFConfig
    from vmtf_net.dataset import SocialFeatureNormalizer, create_dataloaders, create_eval_loader
    from vmtf_net.models.vmtf_net import VMTFNet

logger = logging.getLogger(__name__)

CHECKPOINT_FORMAT_VERSION = 2


def json_default(obj: Any) -> Any:
    """json.dump fallback for numpy scalars / arrays."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if hasattr(obj, "item"):
        return obj.item()
    return str(obj)


# ----------------------------------------------------------------------
# Inference
# ----------------------------------------------------------------------
def move_batch(batch: Dict[str, Any], device: torch.device) -> Dict[str, torch.Tensor]:
    """Map a collated batch to VMTFNet.forward keyword arguments on `device`."""
    inputs = {}
    for key in ("input_ids", "attention_mask", "pixel_values", "social_features", "style_features"):
        if key in batch:
            inputs[key] = batch[key].to(device, non_blocking=True)
    if "has_image" in batch:
        inputs["image_mask"] = batch["has_image"].to(device, non_blocking=True)
    return inputs


@torch.no_grad()
def run_inference(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    use_amp: bool = False,
    criterion: Optional[torch.nn.Module] = None,
    desc: str = "Evaluating",
    disable_tqdm: bool = False,
) -> Dict[str, Any]:
    """
    Run the model over a loader and collect everything needed for metrics and analysis.

    Returns:
        Dict with numpy arrays: 'ids', 'y_true' (None if unlabeled), 'prob_fake',
        'alpha', 'beta', 'cosine', 'has_image' (None when not applicable), 'loss' (float or None),
        'seconds' (wall-clock inference time).
    """
    model.eval()
    ids, labels, probs, alphas, betas, cosines, has_image = [], [], [], [], [], [], []
    total_loss, n_loss_batches = 0.0, 0
    start = time.perf_counter()

    for batch in tqdm(data_loader, desc=desc, leave=False, disable=disable_tqdm, mininterval=5):
        inputs = move_batch(batch, device)
        with torch.amp.autocast(device.type, enabled=use_amp):
            outputs = model(**inputs)

        if criterion is not None and "label" in batch:
            targets = batch["label"].to(device)
            loss, _ = criterion(outputs, targets, inputs.get("image_mask"))
            total_loss += loss.item()
            n_loss_batches += 1

        ids.extend(batch["id"])
        if "label" in batch:
            labels.append(batch["label"])
        probs.append(outputs["probs"][:, 1].float().cpu())
        gate_w = outputs.get("gate_weights") or {}
        if "alpha" in gate_w:
            alphas.append(gate_w["alpha"].squeeze(-1).float().cpu())
            betas.append(gate_w["beta"].squeeze(-1).float().cpu())
        if outputs.get("cosine_sim") is not None:
            cosines.append(outputs["cosine_sim"].float().cpu())
        if "has_image" in batch:
            has_image.append(batch["has_image"])

    def _cat(chunks):
        return torch.cat(chunks).numpy() if chunks else None

    y_true = _cat(labels)
    return {
        "ids": np.array(ids),
        "y_true": y_true.astype(int) if y_true is not None and len(y_true) == len(ids) else None,
        "prob_fake": _cat(probs),
        "alpha": _cat(alphas),
        "beta": _cat(betas),
        "cosine": _cat(cosines),
        "has_image": _cat(has_image),
        "loss": total_loss / n_loss_batches if n_loss_batches else None,
        "seconds": time.perf_counter() - start,
    }


# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------
def compute_metrics(y_true: np.ndarray, prob_fake: np.ndarray, threshold: float = 0.5) -> Dict[str, Any]:
    """Classification metrics at a decision threshold on P(fake)."""
    y_pred = (prob_fake >= threshold).astype(int)
    both_classes = len(np.unique(y_true)) > 1
    return {
        "threshold": float(threshold),
        "n_samples": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_fake": float(f1_score(y_true, y_pred, pos_label=1, average="binary", zero_division=0)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_fake": float(precision_score(y_true, y_pred, pos_label=1, average="binary", zero_division=0)),
        "recall_fake": float(recall_score(y_true, y_pred, pos_label=1, average="binary", zero_division=0)),
        "auc_roc": float(roc_auc_score(y_true, prob_fake)) if both_classes else None,
        "pr_auc": float(average_precision_score(y_true, prob_fake)) if both_classes else None,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }


def find_best_threshold(y_true: np.ndarray, prob_fake: np.ndarray, metric: str = "f1_macro") -> float:
    """Grid-search the decision threshold maximizing `metric` (use on VALIDATION data only)."""
    best_thr, best_val = 0.5, -1.0
    for thr in np.round(np.linspace(0.05, 0.95, 91), 4):
        y_pred = (prob_fake >= thr).astype(int)
        if metric == "f1_fake":
            val = f1_score(y_true, y_pred, pos_label=1, average="binary", zero_division=0)
        else:
            val = f1_score(y_true, y_pred, average="macro", zero_division=0)
        if val > best_val + 1e-9:
            best_thr, best_val = float(thr), val
    return best_thr


def bootstrap_ci(
    y_true: np.ndarray,
    prob_fake: np.ndarray,
    threshold: float,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> Dict[str, Tuple[float, float]]:
    """Percentile bootstrap 95% CIs for Macro-F1, Fake-F1 and AUC-ROC."""
    if n_bootstrap <= 0 or len(y_true) < 2:
        return {}
    rng = np.random.default_rng(seed)
    y_pred = (prob_fake >= threshold).astype(int)
    scores = {"f1_macro": [], "f1_fake": [], "auc_roc": []}
    n = len(y_true)
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        yt, yp, pr = y_true[idx], y_pred[idx], prob_fake[idx]
        scores["f1_macro"].append(f1_score(yt, yp, average="macro", zero_division=0))
        scores["f1_fake"].append(f1_score(yt, yp, pos_label=1, average="binary", zero_division=0))
        if len(np.unique(yt)) > 1:
            scores["auc_roc"].append(roc_auc_score(yt, pr))
    return {
        k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
        for k, v in scores.items()
        if v
    }


def _distribution(values: np.ndarray) -> Dict[str, float]:
    if values is None or len(values) == 0:
        return {}
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "median": float(np.median(values)),
        "p10": float(np.percentile(values, 10)),
        "p90": float(np.percentile(values, 90)),
        "n": int(len(values)),
    }


def interpretability_analysis(out: Dict[str, Any], threshold: float) -> Dict[str, Any]:
    """Gate / cosine distributions per class and metrics on with-image vs without-image subsets."""
    y_true = out["y_true"]
    analysis: Dict[str, Any] = {}
    masks = {"overall": np.ones(len(out["ids"]), dtype=bool)}
    if y_true is not None:
        masks["real"] = y_true == 0
        masks["fake"] = y_true == 1
    with_image = out["has_image"] > 0.5 if out["has_image"] is not None else None

    if out["beta"] is not None:
        analysis["gate_beta_visual"] = {k: _distribution(out["beta"][m]) for k, m in masks.items()}
        if with_image is not None:
            analysis["gate_beta_visual_with_image_only"] = {
                k: _distribution(out["beta"][m & with_image]) for k, m in masks.items()
            }
        hist, edges = np.histogram(out["beta"], bins=10, range=(0.0, 1.0))
        analysis["gate_beta_histogram"] = {"counts": hist.tolist(), "bin_edges": edges.round(2).tolist()}

    if out["cosine"] is not None:
        valid = with_image if with_image is not None else np.ones(len(out["cosine"]), dtype=bool)
        analysis["text_image_cosine"] = {k: _distribution(out["cosine"][m & valid]) for k, m in masks.items()}

    if y_true is not None and with_image is not None:
        subsets = {}
        for name, m in (("with_image", with_image), ("without_image", ~with_image)):
            if m.sum() > 0:
                subsets[name] = compute_metrics(y_true[m], out["prob_fake"][m], threshold)
        analysis["image_availability_subsets"] = subsets

    return analysis


def build_report(
    out: Dict[str, Any],
    threshold: float,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Full evaluation report at the tuned threshold and at the default 0.5 threshold."""
    y_true, prob = out["y_true"], out["prob_fake"]
    report: Dict[str, Any] = {
        "n_samples": int(len(out["ids"])),
        "inference_seconds": float(out["seconds"]),
        "samples_per_second": float(len(out["ids"]) / max(out["seconds"], 1e-9)),
    }
    if y_true is None:
        report["note"] = "Unlabeled data: only predictions were produced."
        return report

    report["metrics"] = compute_metrics(y_true, prob, threshold)
    report["metrics_at_0.5"] = compute_metrics(y_true, prob, 0.5)
    report["confidence_intervals_95"] = bootstrap_ci(y_true, prob, threshold, n_bootstrap, seed)
    y_pred = (prob >= threshold).astype(int)
    report["classification_report"] = classification_report(
        y_true, y_pred, labels=[0, 1], target_names=["Real News (0)", "Fake News (1)"],
        output_dict=True, zero_division=0,
    )
    report["interpretability"] = interpretability_analysis(out, threshold)
    if out["loss"] is not None:
        report["loss"] = float(out["loss"])
    return report


def save_predictions(path: str, out: Dict[str, Any], threshold: float) -> None:
    """Per-sample predictions (one row per post) for error analysis and significance tests."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    df = pd.DataFrame({"id": out["ids"], "prob_fake": out["prob_fake"]})
    df["pred"] = (df["prob_fake"] >= threshold).astype(int)
    if out["y_true"] is not None:
        df["label"] = out["y_true"]
    for key in ("alpha", "beta", "cosine", "has_image"):
        if out[key] is not None:
            df[key] = out[key]
    df.to_csv(path, index=False, encoding="utf-8")


def print_evaluation_summary(results: Dict[str, Any], title: str = "VMTF-NET EVALUATION RESULTS") -> None:
    """Print clean formatted summary to console."""
    print("\n" + "=" * 65)
    print(f"        {title}")
    print("=" * 65)
    m = results.get("metrics")
    if m is None:
        print(f"  {results.get('note', 'No metrics.')}")
        print("=" * 65 + "\n")
        return
    ci = results.get("confidence_intervals_95", {})

    def _ci(key: str) -> str:
        return f"  [95% CI {ci[key][0] * 100:.2f} - {ci[key][1] * 100:.2f}]" if key in ci else ""

    def _pct(v: Optional[float]) -> str:
        return f"{v * 100:.2f}%" if v is not None else "n/a"

    print(f"  Decision threshold:  {m['threshold']:.2f}   (samples: {m['n_samples']})")
    print(f"  Accuracy:            {_pct(m['accuracy'])}")
    print(f"  Macro F1-Score:      {_pct(m['f1_macro'])}{_ci('f1_macro')}")
    print(f"  Fake-class F1:       {_pct(m['f1_fake'])}{_ci('f1_fake')}")
    print(f"  Fake Precision/Rec.: {_pct(m['precision_fake'])} / {_pct(m['recall_fake'])}")
    print(f"  AUC-ROC:             {_pct(m['auc_roc'])}{_ci('auc_roc')}")
    print(f"  PR-AUC:              {_pct(m['pr_auc'])}")
    print("-" * 65)
    print("  Confusion Matrix [[TN, FP], [FN, TP]]:")
    for row in m["confusion_matrix"]:
        print(f"    {row}")
    interp = results.get("interpretability", {})
    gate = interp.get("gate_beta_visual_with_image_only") or interp.get("gate_beta_visual")
    if gate:
        print("-" * 65)
        print("  Visual reliability weight beta (alpha = 1 - beta), samples with images:")
        for k in ("overall", "real", "fake"):
            if gate.get(k):
                print(f"    - {k:<8} mean {gate[k]['mean']:.3f} | median {gate[k]['median']:.3f} | std {gate[k]['std']:.3f}")
    cos = interp.get("text_image_cosine")
    if cos:
        print("  Text-image cosine (shared space):")
        for k in ("real", "fake"):
            if cos.get(k):
                print(f"    - {k:<8} mean {cos[k]['mean']:.3f} | median {cos[k]['median']:.3f}")
    subsets = interp.get("image_availability_subsets", {})
    for name, sm in subsets.items():
        print(f"  Subset {name:<14} n={sm['n_samples']:<5} Macro-F1 {_pct(sm['f1_macro'])}")
    print(f"  Throughput:          {results['samples_per_second']:.1f} samples/s")
    print("=" * 65 + "\n")


# ----------------------------------------------------------------------
# Checkpoint loading
# ----------------------------------------------------------------------
def load_checkpoint(
    checkpoint_path: str,
    device: torch.device,
    text_model_name: Optional[str] = None,
    visual_model_name: Optional[str] = None,
) -> Tuple[VMTFNet, VMTFConfig, Optional[SocialFeatureNormalizer], Dict[str, Any]]:
    """
    Rebuild model + config + training social normalizer from a VMTF checkpoint.
    Backbones are built from their HF config only (weights come from the checkpoint).
    `text_model_name` / `visual_model_name` override the hub ids with local paths (offline Kaggle).
    """
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:  # torch < 1.13
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

    cfg = VMTFConfig.from_dict(checkpoint.get("config", {}))
    if text_model_name:
        cfg.text_model_name = text_model_name
    if visual_model_name:
        cfg.visual_model_name = visual_model_name

    model = VMTFNet(cfg, pretrained_backbones=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    normalizer = None
    if checkpoint.get("social_normalizer") is not None:
        normalizer = SocialFeatureNormalizer.from_state_dict(checkpoint["social_normalizer"])
    return model, cfg, normalizer, checkpoint


def load_style_normalizer(checkpoint: Dict[str, Any]) -> Optional[SocialFeatureNormalizer]:
    """Training-split style-feature normalizer stored in a checkpoint (None if style is unused)."""
    state = checkpoint.get("style_normalizer")
    return SocialFeatureNormalizer.from_state_dict(state) if state is not None else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained VMTF-Net checkpoint")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_vmtf_model.pt", help="Path to checkpoint")
    parser.add_argument("--test_csv", type=str, default="data/test.csv", help="Path to test CSV")
    parser.add_argument("--image_dir", type=str, default="data/images", help="Root directory of images")
    parser.add_argument("--val_csv", type=str, default=None,
                        help="Optional validation CSV used ONLY to tune the decision threshold")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Fixed decision threshold (overrides --val_csv and the checkpoint value)")
    parser.add_argument("--train_csv", type=str, default=None,
                        help="Only for old checkpoints without a stored social normalizer: refit it on train")
    parser.add_argument("--output_json", type=str, default="logs/evaluation_results.json", help="Path to output JSON")
    parser.add_argument("--output_predictions", type=str, default="logs/test_predictions.csv",
                        help="Per-sample predictions CSV")
    parser.add_argument("--batch_size", type=int, default=None, help="Override evaluation batch size")
    parser.add_argument("--num_workers", type=int, default=None, help="Override DataLoader workers")
    parser.add_argument("--text_model_name", type=str, default=None, help="Local path override for the text backbone")
    parser.add_argument("--visual_model_name", type=str, default=None, help="Local path override for the visual backbone")
    parser.add_argument("--n_bootstrap", type=int, default=1000, help="Bootstrap resamples for 95%% CIs (0 to skip)")
    parser.add_argument("--no_tqdm", action="store_true", help="Disable progress bars")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    args = parse_args()

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint file not found: {args.checkpoint}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Loading checkpoint from {args.checkpoint} on {device}")

    # 1. Restore model, config and the TRAINING social normalizer
    model, cfg, normalizer, checkpoint = load_checkpoint(
        args.checkpoint, device, args.text_model_name, args.visual_model_name
    )
    cfg.test_csv = args.test_csv
    cfg.image_dir = args.image_dir
    if args.batch_size:
        cfg.val_batch_size = args.batch_size
    if args.num_workers is not None:
        cfg.num_workers = args.num_workers
    cfg.disable_tqdm = cfg.disable_tqdm or args.no_tqdm
    logger.info(f"Restored experiment: {cfg.experiment_name()} (best val F1: {checkpoint.get('val_metrics', {}).get('f1_macro')})")

    if normalizer is None:
        if not args.train_csv:
            raise ValueError(
                "This checkpoint has no stored social normalizer (old format). "
                "Pass --train_csv so the normalizer is refit on the training split (never on test)."
            )
        logger.warning("Old checkpoint format: refitting social normalizer on --train_csv.")
        train_loader, _, _, normalizer = create_dataloaders(cfg, train_data=args.train_csv)
        style_normalizer = train_loader.dataset.style_normalizer
    else:
        style_normalizer = load_style_normalizer(checkpoint)

    # 2. Decision threshold: CLI > tuned on val > stored in checkpoint > 0.5
    threshold = 0.5
    use_amp = cfg.use_amp and device.type == "cuda"
    if args.threshold is not None:
        threshold = args.threshold
    elif args.val_csv:
        val_loader = create_eval_loader(cfg, args.val_csv, normalizer, style_normalizer=style_normalizer)
        val_out = run_inference(model, val_loader, device, use_amp, desc="Validation", disable_tqdm=cfg.disable_tqdm)
        if val_out["y_true"] is not None:
            threshold = find_best_threshold(val_out["y_true"], val_out["prob_fake"])
    elif checkpoint.get("best_threshold") is not None:
        threshold = float(checkpoint["best_threshold"])
    logger.info(f"Decision threshold: {threshold:.2f}")

    # 3. Test inference
    test_loader = create_eval_loader(cfg, cfg.test_csv, normalizer, style_normalizer=style_normalizer)
    out = run_inference(model, test_loader, device, use_amp, desc="Testing", disable_tqdm=cfg.disable_tqdm)
    results = build_report(out, threshold, n_bootstrap=args.n_bootstrap, seed=cfg.seed)
    results["experiment"] = cfg.experiment_name()
    results["checkpoint"] = os.path.abspath(args.checkpoint)
    print_evaluation_summary(results)

    # 4. Save Results
    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False, default=json_default)
    save_predictions(args.output_predictions, out, threshold)
    logger.info(f"Results saved to: {args.output_json} | predictions: {args.output_predictions}")


if __name__ == "__main__":
    main()
