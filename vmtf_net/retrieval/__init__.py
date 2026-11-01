"""
Evidence / verification layer for VMTF-Net (pure Python + scikit-learn, torch only for the
optional embedding backend):
- ClaimKnowledgeBase : fact-checked claims + similarity search with publication-time cutoff
- SourceCredibility  : trusted / look-alike / unknown domains
- PostChecker        : combines KB matches, sources, style signals and the model probability
"""

from .checker import HINTS, PostChecker
from .claim_kb import ClaimKnowledgeBase, ClaimMatch, normalize_for_matching, to_timestamp
from .source_credibility import DEFAULT_TRUSTED_SOURCES, SourceCredibility, extract_urls, host_of

__all__ = [
    "ClaimKnowledgeBase",
    "ClaimMatch",
    "PostChecker",
    "HINTS",
    "SourceCredibility",
    "DEFAULT_TRUSTED_SOURCES",
    "extract_urls",
    "host_of",
    "normalize_for_matching",
    "to_timestamp",
]
