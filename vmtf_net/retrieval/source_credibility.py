"""
Source credibility: trusted official / mainstream domains and look-alike (spoofed) domains.
==========================================================================================
- `trusted`   : exact domain or sub-domain of a listed source, or any *.gov.vn domain.
- `lookalike` : resembles a trusted domain without being it (typo-squatting such as
                'vnexpres.net', or embedding like 'vnexpress.net.tin-nong.xyz'). A strong warning sign.
- `unknown`   : anything else (NOT a sign of fake news by itself).

The default list is a starting point: edit / extend it with a CSV (columns: domain, tier, name).
Being an official source is evidence, not proof: official outlets also publish corrections.
"""

import logging
import re
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

import pandas as pd

logger = logging.getLogger(__name__)

# domain -> (tier, display name)
DEFAULT_TRUSTED_SOURCES: Dict[str, Tuple[str, str]] = {
    "chinhphu.vn": ("government", "Cổng TTĐT Chính phủ"),
    "baochinhphu.vn": ("government", "Báo Điện tử Chính phủ"),
    "tingia.gov.vn": ("fact_check", "Trung tâm Xử lý tin giả Việt Nam"),
    "vtv.vn": ("national_press", "VTV"),
    "vov.vn": ("national_press", "VOV"),
    "vnanet.vn": ("national_press", "TTXVN"),
    "vietnamplus.vn": ("national_press", "VietnamPlus (TTXVN)"),
    "nhandan.vn": ("national_press", "Báo Nhân Dân"),
    "nhandan.org.vn": ("national_press", "Báo Nhân Dân (tên miền cũ)"),
    "qdnd.vn": ("national_press", "Báo Quân đội Nhân dân"),
    "cand.com.vn": ("national_press", "Báo Công an Nhân dân"),
    "tuoitre.vn": ("press", "Tuổi Trẻ"),
    "thanhnien.vn": ("press", "Thanh Niên"),
    "vnexpress.net": ("press", "VnExpress"),
    "dantri.com.vn": ("press", "Dân Trí"),
    "vietnamnet.vn": ("press", "VietNamNet"),
    "laodong.vn": ("press", "Lao Động"),
    "tienphong.vn": ("press", "Tiền Phong"),
    "znews.vn": ("press", "Znews"),
    "nld.com.vn": ("press", "Người Lao Động"),
    "plo.vn": ("press", "Pháp Luật TP.HCM"),
    "sggp.org.vn": ("press", "Sài Gòn Giải Phóng"),
    "hanoimoi.vn": ("press", "Hànộimới"),
}

_URL_RE = re.compile(r"(?:https?://|www\.)[^\s<>\"')\]]+", re.IGNORECASE)
_BARE_DOMAIN_RE = re.compile(
    r"(?<![\w@.-])((?:[a-z0-9-]+\.)+(?:vn|com|net|org|info|xyz|online|site|news|top|live)(?![\w-])(?:/[^\s<>\"')\]]*)?)",
    re.IGNORECASE,
)


def extract_urls(text: str) -> List[str]:
    """URLs and bare domains ('vnexpress.net/...') found in a text."""
    if not text:
        return []
    found = [u.rstrip(".,;:!?") for u in _URL_RE.findall(text)]
    without_urls = _URL_RE.sub(" ", text)
    found += [m.group(1).rstrip(".,;:!?") for m in _BARE_DOMAIN_RE.finditer(without_urls)]
    return list(dict.fromkeys(found))


def host_of(url: str) -> str:
    """'https://www.Suckhoe.VnExpress.net/a' -> 'suckhoe.vnexpress.net'."""
    u = url.strip()
    if not re.match(r"^[a-z]+://", u, re.IGNORECASE):
        u = "http://" + u
    host = (urlparse(u).hostname or "").lower().rstrip(".")
    return host[4:] if host.startswith("www.") else host


def _levenshtein(a: str, b: str, limit: int = 3) -> int:
    if abs(len(a) - len(b)) > limit:
        return limit + 1
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


class SourceCredibility:
    """Classify domains / URLs as trusted, look-alike (suspicious) or unknown."""

    def __init__(self, trusted: Optional[Dict[str, Tuple[str, str]]] = None, csv_path: Optional[str] = None) -> None:
        self.trusted: Dict[str, Tuple[str, str]] = dict(DEFAULT_TRUSTED_SOURCES if trusted is None else trusted)
        if csv_path:
            df = pd.read_csv(csv_path, dtype=str).fillna("")
            for _, r in df.iterrows():
                domain = host_of(r["domain"])
                if domain:
                    self.trusted[domain] = (r.get("tier", "") or "custom", r.get("name", "") or domain)
            logger.info(f"Loaded {len(df)} trusted sources from {csv_path}")

    def check_domain(self, url_or_domain: str) -> Dict[str, Optional[str]]:
        host = host_of(url_or_domain)
        result = {"input": url_or_domain, "domain": host, "status": "unknown", "tier": None,
                  "name": None, "similar_to": None}
        if not host:
            return result

        for domain, (tier, name) in self.trusted.items():
            if host == domain or host.endswith("." + domain):
                result.update(status="trusted", tier=tier, name=name)
                return result
        if host.endswith(".gov.vn"):
            result.update(status="trusted", tier="government", name=host)
            return result

        # Look-alike checks
        labels = host.split(".")
        tokens = {t for label in labels for t in label.split("-")}  # 'tuoitre-news' -> {'tuoitre', 'news'}
        for domain, (_, name) in self.trusted.items():
            core = domain.split(".")[0]  # 'vnexpress'
            # (a) trusted name embedded in a different domain: vnexpress.net.tin-nong.xyz, tuoitre-news.online
            #     (whole label / hyphen token only: prefixes like 'tuoitrethudo.vn' are other legit outlets)
            if domain in host or (len(core) >= 6 and core in tokens):
                result.update(status="lookalike", similar_to=domain, name=name)
                return result
            # (b) typo-squatting: small edit distance on the whole host or its first label
            if len(core) >= 5:
                if _levenshtein(host, domain) <= 2 or _levenshtein(labels[0], core) == 1:
                    result.update(status="lookalike", similar_to=domain, name=name)
                    return result
        return result

    def check_text(self, text: str) -> List[Dict[str, Optional[str]]]:
        """Credibility of every URL / domain mentioned in a post."""
        return [self.check_domain(u) for u in extract_urls(text)]
