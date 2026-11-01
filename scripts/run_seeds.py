#!/usr/bin/env python3
"""
Multi-seed / K-fold experiment runner for VMTF-Net.
===================================================
Runs `python -m vmtf_net.train` once per (fold, seed), then aggregates every `results.json`
into mean ± std of the test metrics (what a paper table should report).
Re-running the same command is safe: finished runs are skipped, interrupted runs are resumed
from their `last_checkpoint.pt`, which train.py deletes once a run has finished (saves disk).

Examples:
    # 3 seeds of the proposed model (extra train.py arguments go after `--`)
    python scripts/run_seeds.py --seeds 13 42 2024 --out_dir checkpoints/hybrid -- \
        --train_csv data/train.csv --val_csv data/val.csv --test_csv data/test.csv \
        --image_dir data/images --no_tqdm

    # 5-fold cross-validation (folds written by validate_and_preprocess_dataset.py --kfold 5)
    python scripts/run_seeds.py --folds_dir data/folds --seeds 42 --out_dir checkpoints/cv -- \
        --image_dir data/images --fusion_type late_concat --consistency_lambda 0

    # Only aggregate existing runs
    python scripts/run_seeds.py --out_dir checkpoints/hybrid --aggregate_only
"""

import argparse
import glob
import json
import os
import subprocess
import sys
from typing import Any, Dict, List

import numpy as np
import pandas as pd

METRICS = ["accuracy", "f1_macro", "f1_fake", "precision_fake", "recall_fake", "auc_roc", "pr_auc"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run VMTF-Net over several seeds / folds and aggregate results",
        allow_abbrev=False,  # `--seed` meant for train.py must not match `--seeds`
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[13, 42, 2024])
    parser.add_argument("--folds_dir", type=str, default=None, help="Directory with fold_*/{train,val,test}.csv")
    parser.add_argument("--out_dir", type=str, required=True, help="Parent directory for all runs")
    parser.add_argument("--aggregate_only", action="store_true", help="Skip training, only aggregate")
    parser.add_argument("--continue_on_error", action="store_true", help="Keep going if one run fails")
    args, train_args = parser.parse_known_args()
    if train_args and train_args[0] == "--":
        train_args = train_args[1:]
    return args, train_args


def run_all(args, train_args: List[str]) -> None:
    folds = sorted(glob.glob(os.path.join(args.folds_dir, "fold_*"))) if args.folds_dir else [None]
    if args.folds_dir and not folds:
        raise FileNotFoundError(f"No fold_* directories in {args.folds_dir}")

    for fold in folds:
        for seed in args.seeds:
            name = (f"{os.path.basename(fold)}_" if fold else "") + f"seed{seed}"
            run_dir = os.path.join(args.out_dir, name)
            if os.path.exists(os.path.join(run_dir, "results.json")):
                print(f"[skip] {run_dir} already finished")
                continue
            cmd = [sys.executable, "-m", "vmtf_net.train", *train_args, "--seed", str(seed), "--checkpoint_dir", run_dir]
            last_ckpt = os.path.join(run_dir, "last_checkpoint.pt")
            if os.path.exists(last_ckpt):
                # Interrupted run (e.g. Kaggle/Colab timeout): continue where it stopped
                print(f"[resume] {run_dir} from {last_ckpt}")
                cmd += ["--resume", last_ckpt]
            elif os.path.isdir(run_dir) and os.listdir(run_dir):
                # Started but nothing resumable: restart this run from scratch
                cmd += ["--overwrite"]
            if fold:
                cmd += [
                    "--train_csv", os.path.join(fold, "train.csv"),
                    "--val_csv", os.path.join(fold, "val.csv"),
                    "--test_csv", os.path.join(fold, "test.csv"),
                ]
            print(f"\n[run] {' '.join(cmd)}\n", flush=True)
            code = subprocess.call(cmd)
            if code != 0:
                msg = f"Run {name} failed with exit code {code}"
                if not args.continue_on_error:
                    raise RuntimeError(msg)
                print(f"[error] {msg}")


def aggregate(out_dir: str) -> Dict[str, Any]:
    rows = []
    for path in sorted(glob.glob(os.path.join(out_dir, "*", "results.json"))):
        with open(path, "r", encoding="utf-8") as f:
            res = json.load(f)
        metrics = res.get("test", {}).get("metrics")
        if not metrics:
            continue
        row = {"run": os.path.basename(os.path.dirname(path)), "threshold": res.get("threshold"),
               "best_epoch": res.get("best_epoch")}
        row.update({m: metrics.get(m) for m in METRICS})
        rows.append(row)

    if not rows:
        print(f"No finished runs with test metrics under {out_dir}")
        return {}

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "all_runs.csv"), index=False)
    summary = {"n_runs": len(df), "runs": df["run"].tolist()}
    for m in METRICS:
        vals = pd.to_numeric(df[m], errors="coerce").dropna()
        if len(vals):
            summary[m] = {"mean": float(vals.mean()), "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0}
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    print("\n" + "=" * 60)
    print(f"  AGGREGATED TEST RESULTS ({len(df)} runs) - {out_dir}")
    print("=" * 60)
    for m in METRICS:
        if m in summary:
            print(f"  {m:<15} {summary[m]['mean'] * 100:6.2f} ± {summary[m]['std'] * 100:.2f}")
    print("=" * 60)
    return summary


def main():
    args, train_args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    if not args.aggregate_only:
        run_all(args, train_args)
    aggregate(args.out_dir)


if __name__ == "__main__":
    main()
