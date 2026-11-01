#!/usr/bin/env python3
"""
Evidence-layer CLI: check one post, or evaluate the evidence layer on a labeled split.

1) Check one post (what the browser extension backend will do):
    python scripts/check_post.py --kb_dir data/kb/index --text "KHẨN CẤP!!! Chia sẻ gấp ..." \
        --post_time 2024-05-01 [--model_prob 0.83]

2) Evaluate on a labeled split (research table for the evidence layer):
    python scripts/check_post.py --kb_dir data/kb/index --eval_csv data/test.csv --output_json logs/evidence_eval.json

   For each threshold reports:
   - coverage  : % of FAKE posts that match a debunked KB claim  (recall of claim matching)
   - precision : % of posts matching a debunked claim that are really FAKE
   and the AUC of the heuristic clickbait score (how predictive pure style is).
   By default the KB is cut at each post's `timestamp` (--time_cutoff post) to avoid leakage.
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from vmtf_net.retrieval import ClaimKnowledgeBase, PostChecker, SourceCredibility  # noqa: E402
from vmtf_net.style_features import clickbait_score  # noqa: E402


def print_report(report: dict) -> None:
    print("\n" + "=" * 70)
    print(f"  KẾT LUẬN: {report['message']}  [{report['hint']}]")
    if report["model_prob_fake"] is not None:
        print(f"  Xác suất mô hình (tin sai lệch): {report['model_prob_fake'] * 100:.1f}%")
    print(f"  Điểm giật tít (heuristic): {report['clickbait_score']:.2f}")
    for s in report["style_signals"]:
        print(f"    - {s}")
    if report["evidence"]:
        print("  Bằng chứng đối chiếu:")
        for m in report["evidence"]:
            print(f"    [{m['verdict']}] {m['claim_text']}  (score {m['score']:.2f})")
            print(f"        {m['source_name']} - {m['published_at']} - {m['source_url']}")
    if report["related_claims"]:
        print("  Tin liên quan (độ khớp thấp):")
        for m in report["related_claims"]:
            print(f"    [{m['verdict']}] {m['claim_text']}  (score {m['score']:.2f})")
    for u in report["sources_in_post"]:
        extra = f" (giống {u['similar_to']})" if u["similar_to"] else ""
        print(f"  Link trong bài: {u['domain']} -> {u['status']}{extra}")
    print("=" * 70)


def evaluate(kb: ClaimKnowledgeBase, df: pd.DataFrame, time_cutoff: str) -> dict:
    from sklearn.metrics import roc_auc_score

    text_col = next(c for c in ("text", "content", "post_text", "post_message", "title") if c in df.columns)
    labels = pd.to_numeric(df["label"], errors="coerce")
    df = df[labels.isin([0, 1])].reset_index(drop=True)
    y = labels[labels.isin([0, 1])].astype(int).values
    texts = df[text_col].fillna("").astype(str).tolist()

    befores = [None] * len(df)
    if time_cutoff == "post":
        if "timestamp" not in df.columns or df["timestamp"].isna().all():
            print("[warning] No 'timestamp' column: evaluating WITHOUT time cutoff (optimistic, may leak).")
        else:
            befores = df["timestamp"].tolist()

    best = kb.batch_best_scores(texts, befores)
    debunk_scores = np.array([m.score if (m is not None and m.is_debunked) else 0.0 for m in best])

    rows = []
    for thr in [0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 0.9]:
        hit = debunk_scores >= thr
        rows.append({
            "threshold": thr,
            "n_matched": int(hit.sum()),
            "coverage_of_fake": float(hit[y == 1].mean()) if (y == 1).any() else None,
            "precision_fake": float(y[hit].mean()) if hit.any() else None,
            "false_alarms_on_real": int((hit & (y == 0)).sum()),
        })

    cb = np.array([clickbait_score(t) for t in texts])
    result = {
        "n_posts": int(len(y)),
        "n_fake": int(y.sum()),
        "kb_size": len(kb),
        "time_cutoff": time_cutoff,
        "claim_matching": rows,
        "clickbait_auc": float(roc_auc_score(y, cb)) if len(np.unique(y)) > 1 else None,
        "clickbait_mean_real": float(cb[y == 0].mean()) if (y == 0).any() else None,
        "clickbait_mean_fake": float(cb[y == 1].mean()) if (y == 1).any() else None,
    }

    print("\n" + "=" * 70)
    print(f"  EVIDENCE LAYER EVALUATION ({len(y)} posts, {int(y.sum())} fake, KB {len(kb)} claims, cutoff={time_cutoff})")
    print("=" * 70)
    print("  thr    matched  coverage(fake)  precision  false alarms(real)")
    for r in rows:
        cov = f"{r['coverage_of_fake'] * 100:6.1f}%" if r["coverage_of_fake"] is not None else "   n/a"
        prec = f"{r['precision_fake'] * 100:6.1f}%" if r["precision_fake"] is not None else "   n/a"
        print(f"  {r['threshold']:.2f}   {r['n_matched']:>6}   {cov:>12}   {prec:>8}   {r['false_alarms_on_real']:>6}")
    if result["clickbait_auc"] is not None:
        print(f"  Clickbait heuristic AUC: {result['clickbait_auc']:.3f} "
              f"(mean real {result['clickbait_mean_real']:.2f} / fake {result['clickbait_mean_fake']:.2f})")
    print("=" * 70)
    return result


def main():
    parser = argparse.ArgumentParser(description="Check a post against the evidence layer, or evaluate it")
    parser.add_argument("--kb_dir", default=None, help="KB built by scripts/build_claim_kb.py")
    parser.add_argument("--kb_csv", default=None, help="Alternatively build a TF-IDF KB on the fly from a CSV")
    parser.add_argument("--sources_csv", default=None, help="Extra trusted sources (domain,tier,name)")
    parser.add_argument("--text", default=None)
    parser.add_argument("--text_file", default=None)
    parser.add_argument("--post_time", default=None, help="Ignore KB entries published after this date")
    parser.add_argument("--model_prob", type=float, default=None, help="Optional VMTF-Net P(fake)")
    parser.add_argument("--model_threshold", type=float, default=0.5)
    parser.add_argument("--eval_csv", default=None, help="Labeled split to evaluate the evidence layer")
    parser.add_argument("--time_cutoff", default="post", choices=["post", "none"])
    parser.add_argument("--output_json", default=None)
    args = parser.parse_args()

    kb = None
    if args.kb_dir:
        kb = ClaimKnowledgeBase.load(args.kb_dir)
    elif args.kb_csv:
        kb = ClaimKnowledgeBase.from_csv(args.kb_csv)

    if args.eval_csv:
        if kb is None:
            parser.error("--eval_csv requires --kb_dir or --kb_csv")
        result = evaluate(kb, pd.read_csv(args.eval_csv), args.time_cutoff)
    else:
        text = args.text
        if args.text_file:
            with open(args.text_file, encoding="utf-8") as f:
                text = f.read()
        if not text:
            parser.error("Provide --text, --text_file or --eval_csv")
        checker = PostChecker(kb, SourceCredibility(csv_path=args.sources_csv), model_threshold=args.model_threshold)
        result = checker.check(text, post_time=args.post_time, model_prob_fake=args.model_prob)
        print_report(result)

    if args.output_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
