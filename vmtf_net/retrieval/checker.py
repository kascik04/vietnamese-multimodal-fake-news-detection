"""
PostChecker: evidence layer that complements the VMTF-Net probability.
=======================================================================
For one post it reports:
1. Matches in the fact-checked claim KB (debunked / verified claims, with source link and date).
2. Credibility of the URLs / domains mentioned in the post (trusted, look-alike, unknown).
3. Sensational-style signals and a heuristic clickbait score.
4. A cautious hint + Vietnamese message for an end-user UI (browser extension).

It deliberately does NOT output "this is fake": the model detects suspicious *style/patterns*,
the KB only knows claims it contains, and "no evidence found" means "not verified yet".
"""

from typing import Any, Dict, Optional

from ..style_features import clickbait_score, explain_style
from .claim_kb import ClaimKnowledgeBase, DateLike
from .source_credibility import SourceCredibility

HINTS = {
    "matches_debunked": "Nội dung trùng với tin đã bị cơ quan chức năng / báo chính thống bác bỏ.",
    "matches_verified": "Nội dung trùng với thông tin đã được nguồn chính thống xác nhận.",
    "suspicious_source": "Bài dẫn đường link giả mạo tên miền của trang chính thống.",
    "model_suspicious": "Mô hình đánh giá bài có nhiều dấu hiệu của tin sai lệch; chưa tìm thấy bằng chứng đối chiếu.",
    "clickbait_style": "Bài có văn phong giật tít; chưa tìm thấy bằng chứng đối chiếu.",
    "no_evidence": "Chưa tìm thấy thông tin đối chiếu. Hãy kiểm tra thêm từ nguồn chính thống.",
}


class PostChecker:
    def __init__(
        self,
        kb: Optional[ClaimKnowledgeBase] = None,
        sources: Optional[SourceCredibility] = None,
        model_threshold: float = 0.5,
        clickbait_threshold: float = 0.6,
    ) -> None:
        """
        Args:
            kb: fact-checked claim knowledge base (optional).
            sources: domain credibility checker (default trusted list if None).
            model_threshold: decision threshold of VMTF-Net (use the val-tuned `best_threshold`).
            clickbait_threshold: heuristic clickbait score above which the style is flagged.
        """
        self.kb = kb
        self.sources = sources or SourceCredibility()
        self.model_threshold = model_threshold
        self.clickbait_threshold = clickbait_threshold

    def check(
        self,
        text: str,
        post_time: DateLike = None,
        model_prob_fake: Optional[float] = None,
        top_k: int = 3,
    ) -> Dict[str, Any]:
        """
        Args:
            text: post text.
            post_time: when the post was published / checked; KB entries published later are ignored.
            model_prob_fake: optional P(fake) from VMTF-Net for the same post.
            top_k: number of KB matches to return.
        """
        matches = self.kb.search(text, top_k=top_k, before=post_time) if self.kb is not None else []
        strong = [m for m in matches if self.kb is not None and m.score >= self.kb.threshold]
        urls = self.sources.check_text(text)
        cb_score = clickbait_score(text)

        if any(m.is_debunked for m in strong):
            hint = "matches_debunked"
        elif any(m.verdict == "true" for m in strong):
            hint = "matches_verified"
        elif any(u["status"] == "lookalike" for u in urls):
            hint = "suspicious_source"
        elif model_prob_fake is not None and model_prob_fake >= self.model_threshold:
            hint = "model_suspicious"
        elif cb_score >= self.clickbait_threshold:
            hint = "clickbait_style"
        else:
            hint = "no_evidence"

        return {
            "hint": hint,
            "message": HINTS[hint],
            "model_prob_fake": model_prob_fake,
            "evidence": [m.to_dict() for m in strong],
            # Weaker but still plausible matches (at least half the match threshold), for manual review
            "related_claims": [
                m.to_dict() for m in matches if m not in strong and m.score >= 0.5 * self.kb.threshold
            ],
            "sources_in_post": urls,
            "clickbait_score": round(cb_score, 3),
            "style_signals": explain_style(text),
        }
