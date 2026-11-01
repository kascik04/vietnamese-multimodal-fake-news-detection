"""
Knowledge base of fact-checked claims + similarity search ("this post looks like a claim
that was already debunked on <date> by <official source>").
===========================================================================================
KB CSV columns (see data/kb/debunked_claims_template.csv):
    claim_id, claim_text, verdict, source_name, source_url, published_at, topic, notes
    verdict in {false, misleading, true, unverified}

Backends:
- "tfidf"     : char 3-5-gram + word 1-2-gram TF-IDF (offline, no download, robust to typos and
                missing diacritics). Default.
- "embedding" : any HuggingFace sentence encoder with mean pooling (paraphrase-level matching,
                e.g. sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2). Needs torch.

Long posts are split into overlapping sentence windows and the best window score is kept, so a
short claim is not diluted by the rest of the post.

ANTI-LEAKAGE: `search(..., before=post_time)` only returns KB entries published BEFORE that time.
When evaluating on a test set, always pass the post timestamp; otherwise the debunk article of
the test post itself is found and results are unrealistically good.
"""

import json
import logging
import os
import pickle
import re
import unicodedata
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

KB_COLUMNS: List[str] = [
    "claim_id", "claim_text", "verdict", "source_name", "source_url", "published_at", "topic", "notes",
]
VALID_VERDICTS = {"false", "misleading", "true", "unverified"}
DEBUNKED_VERDICTS = {"false", "misleading"}

# Score above which a match is considered "the same claim" (calibrate with scripts/check_post.py --eval_csv)
DEFAULT_THRESHOLDS = {"tfidf": 0.45, "embedding": 0.75}

DateLike = Union[str, pd.Timestamp, None]


def normalize_for_matching(text: Any) -> str:
    """Lowercase, NFC, remove URLs / hashtags signs / punctuation, collapse spaces (keeps diacritics)."""
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return ""
    t = unicodedata.normalize("NFC", str(text)).lower()
    t = re.sub(r"https?://\S+|www\.\S+", " ", t)
    t = re.sub(r"[^\w\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def split_into_windows(text: str, window: int = 2, max_windows: int = 30) -> List[str]:
    """Whole text + sliding windows of `window` sentences (for short-claim vs long-post matching)."""
    text = "" if text is None else str(text)
    sentences = [s.strip() for s in re.split(r"(?<=[.!?…\n])\s+", text) if len(s.strip()) > 10]
    chunks = [text]
    if len(sentences) > 1:
        for i in range(len(sentences)):
            chunks.append(" ".join(sentences[i:i + window]))
            if len(chunks) > max_windows:
                break
    return chunks


def to_timestamp(value: DateLike) -> Optional[pd.Timestamp]:
    """
    Parse a date/time to a naive Timestamp. ISO 'YYYY-MM-DD[ HH:MM]' is recommended;
    'dd/mm/yyyy' (Vietnamese convention) is read day-first. Returns None when missing/invalid.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)) or str(value).strip() == "":
        return None
    if isinstance(value, pd.Timestamp):
        ts = value
    else:
        s = str(value).strip()
        day_first = bool(re.match(r"^\d{1,2}[/.-]\d{1,2}[/.-]\d{4}", s))
        ts = pd.to_datetime(s, errors="coerce", dayfirst=day_first)
    if pd.isna(ts):
        return None
    return ts.tz_localize(None) if ts.tzinfo is not None else ts


@dataclass
class ClaimMatch:
    claim_id: str
    claim_text: str
    verdict: str
    source_name: str
    source_url: str
    published_at: Optional[str]
    score: float
    matched_text: str

    @property
    def is_debunked(self) -> bool:
        return self.verdict in DEBUNKED_VERDICTS

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["is_debunked"] = self.is_debunked
        return d


class _TfidfIndex:
    def __init__(self) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=1)
        self.word = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), sublinear_tf=True, min_df=1,
                                    token_pattern=r"(?u)\b\w+\b")
        self.char_matrix = None
        self.word_matrix = None

    def fit(self, texts: List[str]) -> None:
        self.char_matrix = self.char.fit_transform(texts)
        self.word_matrix = self.word.fit_transform(texts)

    def scores(self, queries: List[str]) -> np.ndarray:
        """[n_queries, n_claims] cosine scores (TF-IDF rows are L2-normalized)."""
        c = (self.char.transform(queries) @ self.char_matrix.T).toarray()
        w = (self.word.transform(queries) @ self.word_matrix.T).toarray()
        return 0.5 * c + 0.5 * w


class _EmbeddingIndex:
    def __init__(self, model_name: str, device: Optional[str] = None, batch_size: int = 32) -> None:
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self._model = None
        self._tokenizer = None
        self.matrix: Optional[np.ndarray] = None

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModel.from_pretrained(self.model_name).to(self.device).eval()

    def encode(self, texts: List[str]) -> np.ndarray:
        import torch

        self._load()
        out = []
        with torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                enc = self._tokenizer(texts[i:i + self.batch_size], padding=True, truncation=True,
                                      max_length=256, return_tensors="pt").to(self.device)
                hidden = self._model(**enc).last_hidden_state
                mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                emb = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1.0)
                out.append(torch.nn.functional.normalize(emb, dim=-1).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 1), dtype=np.float32)

    def fit(self, texts: List[str]) -> None:
        self.matrix = self.encode(texts)

    def scores(self, queries: List[str]) -> np.ndarray:
        return self.encode(queries) @ self.matrix.T


class ClaimKnowledgeBase:
    """Fact-checked claims with similarity search and an optional publication-time cutoff."""

    def __init__(
        self,
        claims: pd.DataFrame,
        backend: str = "tfidf",
        embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        threshold: Optional[float] = None,
        device: Optional[str] = None,
    ) -> None:
        if backend not in DEFAULT_THRESHOLDS:
            raise ValueError(f"backend must be one of {list(DEFAULT_THRESHOLDS)}")
        self.claims = self._validate(claims)
        self.backend = backend
        self.embedding_model = embedding_model
        self.threshold = threshold if threshold is not None else DEFAULT_THRESHOLDS[backend]
        self._published = [to_timestamp(v) for v in self.claims["published_at"]]
        self.index: Union[_TfidfIndex, _EmbeddingIndex, None] = None
        self.device = device

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    @staticmethod
    def _validate(df: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in ("claim_id", "claim_text", "verdict") if c not in df.columns]
        if missing:
            raise KeyError(f"KB CSV is missing required columns: {missing}")
        df = df.copy()
        for c in KB_COLUMNS:
            if c not in df.columns:
                df[c] = ""
        df = df[KB_COLUMNS].fillna("")
        df["claim_id"] = df["claim_id"].astype(str)
        df["verdict"] = df["verdict"].astype(str).str.strip().str.lower()
        bad = ~df["verdict"].isin(VALID_VERDICTS)
        if bad.any():
            logger.warning(f"Dropping {int(bad.sum())} KB rows with verdict not in {sorted(VALID_VERDICTS)}.")
            df = df[~bad]
        empty = df["claim_text"].astype(str).str.strip() == ""
        if empty.any():
            df = df[~empty]
        dup = df["claim_id"].duplicated()
        if dup.any():
            logger.warning(f"Dropping {int(dup.sum())} KB rows with duplicated claim_id.")
            df = df[~dup]
        return df.reset_index(drop=True)

    @classmethod
    def from_csv(cls, path: str, keep_examples: bool = False, **kwargs) -> "ClaimKnowledgeBase":
        df = pd.read_csv(path, dtype=str)
        if not keep_examples and "claim_id" in df.columns:
            df = df[~df["claim_id"].astype(str).str.startswith("EXAMPLE_")]
        kb = cls(df, **kwargs)
        kb.build()
        return kb

    def build(self) -> "ClaimKnowledgeBase":
        if len(self.claims) == 0:
            raise ValueError("The knowledge base is empty.")
        texts = [normalize_for_matching(t) for t in self.claims["claim_text"]]
        if self.backend == "tfidf":
            self.index = _TfidfIndex()
        else:
            self.index = _EmbeddingIndex(self.embedding_model, self.device)
        self.index.fit(texts)
        logger.info(f"Claim KB built: {len(self.claims)} claims, backend={self.backend}.")
        return self

    def __len__(self) -> int:
        return len(self.claims)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------
    def _available(self, before: DateLike) -> np.ndarray:
        """Mask of KB entries usable at time `before` (entries without a date are excluded then)."""
        if before is None:
            return np.ones(len(self.claims), dtype=bool)
        cutoff = to_timestamp(before)
        if cutoff is None:
            return np.ones(len(self.claims), dtype=bool)
        return np.array([p is not None and p < cutoff for p in self._published], dtype=bool)

    def search(
        self,
        text: str,
        top_k: int = 5,
        before: DateLike = None,
        min_score: Optional[float] = 0.0,
    ) -> List[ClaimMatch]:
        """
        Most similar KB claims to `text`.

        Args:
            text: post text.
            top_k: number of matches returned.
            before: only KB entries published strictly before this date (anti-leakage / realism).
            min_score: drop matches below this score (None = keep all).
        """
        if self.index is None:
            self.build()
        windows = split_into_windows(text)
        queries = [normalize_for_matching(w) for w in windows]
        scores = self.index.scores(queries)               # [n_windows, n_claims]
        best_window = scores.argmax(axis=0)                # best window per claim
        best_score = scores.max(axis=0)
        best_score = np.where(self._available(before), best_score, -1.0)

        order = np.argsort(-best_score)[:top_k]
        matches = []
        for j in order:
            s = float(best_score[j])
            if s < 0 or (min_score is not None and s < min_score):
                continue
            row = self.claims.iloc[j]
            matches.append(ClaimMatch(
                claim_id=row["claim_id"],
                claim_text=row["claim_text"],
                verdict=row["verdict"],
                source_name=row["source_name"],
                source_url=row["source_url"],
                published_at=row["published_at"] or None,
                score=round(s, 4),
                matched_text=windows[int(best_window[j])][:300],
            ))
        return matches

    def best_match(self, text: str, before: DateLike = None) -> Optional[ClaimMatch]:
        """Top match if its score reaches the KB threshold, else None."""
        matches = self.search(text, top_k=1, before=before)
        return matches[0] if matches and matches[0].score >= self.threshold else None

    def batch_best_scores(
        self, texts: Sequence[str], befores: Optional[Sequence[DateLike]] = None
    ) -> List[Optional[ClaimMatch]]:
        """Top match (regardless of threshold) for many posts, e.g. for evaluation."""
        befores = befores if befores is not None else [None] * len(texts)
        out = []
        for t, b in zip(texts, befores):
            m = self.search(t, top_k=1, before=b)
            out.append(m[0] if m else None)
        return out

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        self.claims.to_csv(os.path.join(directory, "claims.csv"), index=False, encoding="utf-8")
        meta = {"backend": self.backend, "embedding_model": self.embedding_model, "threshold": self.threshold}
        with open(os.path.join(directory, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        if self.backend == "tfidf":
            with open(os.path.join(directory, "tfidf.pkl"), "wb") as f:
                pickle.dump(self.index, f)
        else:
            np.save(os.path.join(directory, "embeddings.npy"), self.index.matrix)

    @classmethod
    def load(cls, directory: str, device: Optional[str] = None) -> "ClaimKnowledgeBase":
        """Load a KB saved with `save()` (only load directories you created yourself: pickle)."""
        with open(os.path.join(directory, "meta.json"), encoding="utf-8") as f:
            meta = json.load(f)
        claims = pd.read_csv(os.path.join(directory, "claims.csv"), dtype=str).fillna("")
        kb = cls(claims, backend=meta["backend"], embedding_model=meta["embedding_model"],
                 threshold=meta["threshold"], device=device)
        if kb.backend == "tfidf":
            with open(os.path.join(directory, "tfidf.pkl"), "rb") as f:
                kb.index = pickle.load(f)
        else:
            kb.index = _EmbeddingIndex(kb.embedding_model, device)
            kb.index.matrix = np.load(os.path.join(directory, "embeddings.npy"))
        return kb
