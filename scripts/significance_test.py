#!/usr/bin/env python3
"""
Statistical significance test between two models evaluated on the SAME test set.
================================================================================
Inputs are the per-sample prediction CSVs written by training/evaluation
(`test_predictions.csv`: columns id, label, prob_fake, pred).

Reports:
- McNemar's exact test on paired correct/incorrect decisions.
- Paired bootstrap of the difference (B - A) in Macro-F1 and AUC-ROC with 95% CI and p-value.

Example:
    python scripts/significance_test.py \
        --a checkpoints/late_concat_seed42/test_predictions.csv --name_a late_concat \
        --b checkpoints/hybrid_seed42/test_predictions.csv --name_b hybrid_proposed
"""

import argparse
import json
from math import comb

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score


def mcnemar_exact(correct_a: np.ndarray, correct_b: np.ndarray) -> dict:
    """Two-sided exact McNemar test (binomial on discordant pairs)."""
    b01 = int(np.sum(correct_a & ~correct_b))  # A right, B wrong
    b10 = int(np.sum(~correct_a & correct_b))  # A wrong, B right
    n = b01 + b10
    if n == 0:
        return {"a_only_correct": b01, "b_only_correct": b10, "p_value": 1.0}
    k = min(b01, b10)
    p = sum(comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return {"a_only_correct": b01, "b_only_correct": b10, "p_value": float(min(1.0, 2 * p))}


def paired_bootstrap(y, prob_a, pred_a, prob_b, pred_b, n_bootstrap: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    n = len(y)
    diffs = {"f1_macro": [], "auc_roc": []}
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        yt = y[idx]
        diffs["f1_macro"].append(
            f1_score(yt, pred_b[idx], average="macro", zero_division=0)
            - f1_score(yt, pred_a[idx], average="macro", zero_division=0)
        )
        if len(np.unique(yt)) > 1:
            diffs["auc_roc"].append(roc_auc_score(yt, prob_b[idx]) - roc_auc_score(yt, prob_a[idx]))
    out = {}
    for k, v in diffs.items():
        if not v:
            continue
        v = np.asarray(v)
        # two-sided p-value: how often the sign of the difference flips
        p = 2 * min(np.mean(v <= 0), np.mean(v >= 0))
        out[k] = {
            "mean_diff_b_minus_a": float(v.mean()),
            "ci95": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))],
            "p_value": float(min(1.0, p)),
        }
    return out


def main():
    parser = argparse.ArgumentParser(description="McNemar + paired bootstrap between two prediction files")
    parser.add_argument("--a", required=True, help="Predictions CSV of model A (baseline)")
    parser.add_argument("--b", required=True, help="Predictions CSV of model B (proposed)")
    parser.add_argument("--name_a", default="A")
    parser.add_argument("--name_b", default="B")
    parser.add_argument("--n_bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_json", default=None)
    args = parser.parse_args()

    a = pd.read_csv(args.a, dtype={"id": str})
    b = pd.read_csv(args.b, dtype={"id": str})
    merged = a.merge(b, on="id", suffixes=("_a", "_b"))
    if len(merged) != len(a) or len(merged) != len(b):
        print(f"[warning] {len(a)} / {len(b)} rows, {len(merged)} matched by id: the test sets differ!")
    if (merged["label_a"] != merged["label_b"]).any():
        raise ValueError("Labels differ between files for the same ids.")

    y = merged["label_a"].values.astype(int)
    pred_a, pred_b = merged["pred_a"].values.astype(int), merged["pred_b"].values.astype(int)
    prob_a, prob_b = merged["prob_fake_a"].values, merged["prob_fake_b"].values

    result = {
        "n_samples": int(len(y)),
        args.name_a: {"f1_macro": float(f1_score(y, pred_a, average="macro", zero_division=0))},
        args.name_b: {"f1_macro": float(f1_score(y, pred_b, average="macro", zero_division=0))},
        "mcnemar": mcnemar_exact(pred_a == y, pred_b == y),
        "paired_bootstrap": paired_bootstrap(y, prob_a, pred_a, prob_b, pred_b, args.n_bootstrap, args.seed),
    }

    print(json.dumps(result, indent=4))
    p = result["mcnemar"]["p_value"]
    verdict = "significant" if p < 0.05 else "NOT significant"
    print(f"\nMcNemar p = {p:.4f} -> difference between {args.name_b} and {args.name_a} is {verdict} at alpha = 0.05")
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=4)


if __name__ == "__main__":
    main()
