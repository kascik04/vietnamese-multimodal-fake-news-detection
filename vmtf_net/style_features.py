"""
Stylistic / clickbait features for Vietnamese social-media posts.
=================================================================
Hand-crafted, interpretable signals of sensational writing ("SỐC!!!", "chia sẻ gấp", ⚠️🔥):
- used as an extra model branch (`--use_style`, normalized with statistics fitted on train),
- used to explain a prediction to end users (`explain_style`),
- used as a fast heuristic pre-filter (`clickbait_score`).

IMPORTANT: clickbait style is NOT proof of fake news (mainstream outlets also use sensational
titles). These features are signals to combine with content and evidence, never a verdict.
Pure Python (no torch), so it also runs inside a lightweight API / extension backend.
"""

import re
import unicodedata
from typing import Dict, List

import numpy as np
import pandas as pd

STYLE_FEATURE_NAMES: List[str] = [
    "char_len",
    "word_len",
    "exclamation_count",
    "question_count",
    "repeated_punct_count",
    "uppercase_ratio",
    "all_caps_word_ratio",
    "emoji_count",
    "alarm_emoji_count",
    "clickbait_term_count",
    "share_request_count",
    "url_count",
    "hashtag_count",
    "number_count",
]

# Sensational / clickbait vocabulary frequently seen in Vietnamese misinformation
CLICKBAIT_TERMS: List[str] = [
    "sốc", "cực sốc", "khẩn cấp", "cấp báo", "cảnh báo", "tin nóng", "nóng hổi", "bất ngờ",
    "kinh hoàng", "rùng mình", "chấn động", "gây bão", "xôn xao", "hoang mang", "sự thật",
    "không thể tin", "không ai ngờ", "chưa từng có", "bí mật", "tiết lộ", "lộ clip",
    "thần dược", "chữa khỏi", "khỏi hẳn", "bách bệnh", "100%", "miễn phí", "trúng thưởng",
    "nhận quà", "đừng bỏ qua", "ai cũng nên biết", "hot",
]

# Explicit calls to spread the message
SHARE_TERMS: List[str] = [
    "chia sẻ", "share", "lan tỏa", "lan toả", "cho mọi người cùng biết", "cho mọi người biết",
    "đọc ngay", "xem ngay", "báo cho người thân", "gửi cho",
]

ALARM_EMOJIS = "⚠🔥‼❗❕🚨📢📣⛔🆘❌😱😨💥"

_EMOJI_RE = re.compile(
    "[\U0001F300-\U0001F5FF\U0001F600-\U0001F64F\U0001F680-\U0001F6FF\U0001F900-\U0001F9FF"
    "\U0001FA70-\U0001FAFF☀-➿⬀-⯿‼⁉]"
)
_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_HASHTAG_RE = re.compile(r"(?<!\w)#\w+")
_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)*")
_REPEATED_PUNCT_RE = re.compile(r"[!?]{2,}|\.{3,}|…")
_WORD_RE = re.compile(r"\w+", re.UNICODE)


def strip_accents(text: str) -> str:
    """'Khẩn cấp' -> 'Khan cap' (also maps đ/Đ to d/D)."""
    text = text.replace("đ", "d").replace("Đ", "D")
    return "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")


def _term_patterns(terms: List[str]) -> List[re.Pattern]:
    return [re.compile(rf"(?<!\w){re.escape(t)}(?!\w)") for t in terms]


_CLICKBAIT_PATTERNS = _term_patterns(CLICKBAIT_TERMS)
_CLICKBAIT_PATTERNS_ASCII = _term_patterns([strip_accents(t) for t in CLICKBAIT_TERMS])
_SHARE_PATTERNS = _term_patterns(SHARE_TERMS)
_SHARE_PATTERNS_ASCII = _term_patterns([strip_accents(t) for t in SHARE_TERMS])


def _is_unaccented(text: str) -> bool:
    """True for text typed without Vietnamese diacritics ('khan cap chia se gap')."""
    letters = [c for c in text if c.isalpha()]
    if len(letters) < 20:
        return False
    return sum(1 for c in letters if ord(c) > 127) / len(letters) < 0.02


def _find_terms(text: str, terms: List[str], patterns, ascii_patterns) -> List[str]:
    lower = unicodedata.normalize("NFC", text.lower())
    if _is_unaccented(lower):
        return [t for t, p in zip(terms, ascii_patterns) if p.search(lower)]
    return [t for t, p in zip(terms, patterns) if p.search(lower)]


def extract_style_features(text: str) -> Dict[str, float]:
    """All features are non-negative counts or ratios (log1p + standardization is applied later)."""
    text = "" if text is None else str(text)
    text = unicodedata.normalize("NFC", text)
    no_url = _URL_RE.sub(" ", text)
    words = _WORD_RE.findall(no_url)
    letters = [c for c in no_url if c.isalpha()]
    caps_words = [w for w in words if len(w) >= 2 and w.isalpha() and w.isupper()]

    return {
        "char_len": float(len(text)),
        "word_len": float(len(words)),
        "exclamation_count": float(text.count("!")),
        "question_count": float(text.count("?")),
        "repeated_punct_count": float(len(_REPEATED_PUNCT_RE.findall(text))),
        "uppercase_ratio": (sum(c.isupper() for c in letters) / len(letters)) if letters else 0.0,
        "all_caps_word_ratio": (len(caps_words) / len(words)) if words else 0.0,
        "emoji_count": float(len(_EMOJI_RE.findall(text))),
        "alarm_emoji_count": float(sum(text.count(e) for e in ALARM_EMOJIS)),
        "clickbait_term_count": float(len(_find_terms(no_url, CLICKBAIT_TERMS, _CLICKBAIT_PATTERNS, _CLICKBAIT_PATTERNS_ASCII))),
        "share_request_count": float(len(_find_terms(no_url, SHARE_TERMS, _SHARE_PATTERNS, _SHARE_PATTERNS_ASCII))),
        "url_count": float(len(_URL_RE.findall(text))),
        "hashtag_count": float(len(_HASHTAG_RE.findall(no_url))),
        "number_count": float(len(_NUMBER_RE.findall(no_url))),
    }


def extract_style_dataframe(texts: List[str]) -> pd.DataFrame:
    """Style features of many texts as a DataFrame with columns STYLE_FEATURE_NAMES."""
    rows = [extract_style_features(t) for t in texts]
    return pd.DataFrame(rows, columns=STYLE_FEATURE_NAMES).astype(np.float64)


def clickbait_score(text: str) -> float:
    """
    Heuristic sensationalism score in [0, 1] (NOT a fake-news probability, not trained).
    Useful for fast triage and as a baseline to report against the learned model.
    """
    f = extract_style_features(text)
    raw = (
        1.0 * min(f["clickbait_term_count"], 3)
        + 1.0 * min(f["share_request_count"], 2)
        + 0.5 * min(f["exclamation_count"], 4)
        + 0.5 * min(f["repeated_punct_count"], 2)
        + 0.8 * min(f["alarm_emoji_count"], 3)
        + 3.0 * min(f["all_caps_word_ratio"], 0.5)
    )
    return float(1.0 - np.exp(-raw / 3.0))


def explain_style(text: str) -> List[str]:
    """Human-readable (Vietnamese) list of sensational-style signals found in a post."""
    text = "" if text is None else str(text)
    f = extract_style_features(text)
    no_url = _URL_RE.sub(" ", text)
    signals = []
    terms = _find_terms(no_url, CLICKBAIT_TERMS, _CLICKBAIT_PATTERNS, _CLICKBAIT_PATTERNS_ASCII)
    if terms:
        signals.append("Từ ngữ giật tít: " + ", ".join(terms))
    shares = _find_terms(no_url, SHARE_TERMS, _SHARE_PATTERNS, _SHARE_PATTERNS_ASCII)
    if shares:
        signals.append("Kêu gọi lan truyền: " + ", ".join(shares))
    if f["exclamation_count"] >= 2:
        signals.append(f"Nhiều dấu chấm than ({int(f['exclamation_count'])})")
    if f["repeated_punct_count"] >= 1:
        signals.append("Dấu câu lặp lại (!!!, ???, ...)")
    if f["all_caps_word_ratio"] >= 0.2:
        signals.append(f"Viết IN HOA nhiều ({f['all_caps_word_ratio'] * 100:.0f}% số từ)")
    if f["alarm_emoji_count"] >= 1:
        found = "".join(sorted({e for e in ALARM_EMOJIS if e in text}))
        signals.append(f"Biểu tượng báo động: {found}")
    return signals
