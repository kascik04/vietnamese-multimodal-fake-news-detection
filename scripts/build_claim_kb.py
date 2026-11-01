#!/usr/bin/env python3
"""
Build (and save) the fact-checked claim knowledge base used by the evidence layer.

CSV columns: claim_id, claim_text, verdict (false|misleading|true|unverified), source_name,
source_url, published_at (YYYY-MM-DD), topic, notes  -> see data/kb/debunked_claims_template.csv

How to fill it (one row per claim, NOT per post):
- false / misleading: claims officially refuted (tingia.gov.vn, police / provincial portals,
  mainstream press articles "bác bỏ tin đồn ..."). source_url = the refutation article.
- true: official announcements that are often mis-reported, to recognise accurate posts.
- published_at = date of the refutation / announcement (used for the anti-leakage time cutoff).

Examples:
    python scripts/build_claim_kb.py --kb_csv data/kb/debunked_claims.csv --out_dir data/kb/index
    python scripts/build_claim_kb.py --kb_csv data/kb/debunked_claims.csv --out_dir data/kb/index_emb \
        --backend embedding --embedding_model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
"""

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from vmtf_net.retrieval import ClaimKnowledgeBase  # noqa: E402


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="Build the fact-checked claim knowledge base index")
    parser.add_argument("--kb_csv", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--backend", default="tfidf", choices=["tfidf", "embedding"])
    parser.add_argument("--embedding_model", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    parser.add_argument("--threshold", type=float, default=None, help="Match threshold (default per backend)")
    parser.add_argument("--keep_examples", action="store_true", help="Keep EXAMPLE_ rows (testing only)")
    args = parser.parse_args()

    kb = ClaimKnowledgeBase.from_csv(
        args.kb_csv, keep_examples=args.keep_examples, backend=args.backend,
        embedding_model=args.embedding_model, threshold=args.threshold,
    )
    kb.save(args.out_dir)
    counts = kb.claims["verdict"].value_counts().to_dict()
    print(f"Saved KB with {len(kb)} claims {counts} to {args.out_dir} (threshold {kb.threshold})")


if __name__ == "__main__":
    main()
