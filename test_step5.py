"""
OFFLINE tests for the style features and the evidence layer (no torch required).
Validates:
1. Style / clickbait features, unaccented text handling, explanations, heuristic score.
2. Claim knowledge base: paraphrase / typo matching, long-post windows, time cutoff (anti-leakage),
   verdict filtering, save/load round-trip.
3. Source credibility: trusted, sub-domain, *.gov.vn, look-alike (typo-squatting / embedding), unknown.
4. PostChecker hints and report structure.
5. scripts/check_post.py evaluation function on a tiny labeled split.
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import pandas as pd

from vmtf_net.retrieval import ClaimKnowledgeBase, PostChecker, SourceCredibility, extract_urls
from vmtf_net.style_features import (
    STYLE_FEATURE_NAMES,
    clickbait_score,
    explain_style,
    extract_style_dataframe,
    extract_style_features,
)

CLICKBAIT = "⚠️ KHẨN CẤP!!! Chia sẻ gấp cho mọi người cùng biết: uống nước chanh nóng diệt sạch virus 100% 🔥🔥"
NEUTRAL = "Ngân hàng Nhà nước thông báo điều chỉnh lãi suất điều hành, áp dụng từ ngày 1/9."


def test_style() -> None:
    print("\n[1] Style / clickbait features...")
    f = extract_style_features(CLICKBAIT)
    assert set(f) == set(STYLE_FEATURE_NAMES)
    assert all(v >= 0 for v in f.values()), "Features must be non-negative (log1p later)"
    assert f["exclamation_count"] == 3 and f["alarm_emoji_count"] >= 3
    assert f["clickbait_term_count"] >= 2 and f["share_request_count"] >= 1
    assert f["all_caps_word_ratio"] > 0

    n = extract_style_features(NEUTRAL)
    assert n["clickbait_term_count"] == 0 and n["exclamation_count"] == 0
    assert clickbait_score(CLICKBAIT) > 0.7 > 0.2 > clickbait_score(NEUTRAL)

    # Unaccented typing must still be detected
    unaccented = "khan cap chia se gap cho moi nguoi, uong nuoc chanh nong diet virus ngay lap tuc"
    assert extract_style_features(unaccented)["clickbait_term_count"] >= 1
    assert extract_style_features(unaccented)["share_request_count"] >= 1

    signals = explain_style(CLICKBAIT)
    print("    signals:", signals)
    assert any("giật tít" in s for s in signals) and any("lan truyền" in s for s in signals)
    assert explain_style(NEUTRAL) == []

    df = extract_style_dataframe([CLICKBAIT, NEUTRAL, "", None])
    assert list(df.columns) == STYLE_FEATURE_NAMES and len(df) == 4
    print("    [OK] style features passed.")


def make_kb_df() -> pd.DataFrame:
    return pd.DataFrame([
        {"claim_id": "C1", "claim_text": "Uống nước chanh nóng có thể tiêu diệt virus corona trong cơ thể",
         "verdict": "false", "source_name": "Bộ Y tế", "source_url": "https://moh.gov.vn/x",
         "published_at": "2020-03-10"},
        {"claim_id": "C2", "claim_text": "Phát hiện sinh vật lạ khổng lồ dạt vào bờ biển miền Trung",
         "verdict": "misleading", "source_name": "Báo A", "source_url": "https://tuoitre.vn/y",
         "published_at": "15/06/2024"},
        {"claim_id": "C3", "claim_text": "Ngân hàng Nhà nước điều chỉnh lãi suất điều hành",
         "verdict": "true", "source_name": "SBV", "source_url": "https://sbv.gov.vn/z",
         "published_at": "2024-08-20"},
        {"claim_id": "C4", "claim_text": "Dòng có verdict sai phải bị loại", "verdict": "maybe"},
    ])


def test_claim_kb(tmp: str) -> ClaimKnowledgeBase:
    print("\n[2] Claim knowledge base...")
    kb = ClaimKnowledgeBase(make_kb_df()).build()
    assert len(kb) == 3, "Row with invalid verdict must be dropped"

    # Paraphrase with typos / no diacritics, inside a long post
    post = ("Mọi người ơi đọc kỹ nhé. Hôm qua tôi đi chợ gặp nhiều chuyện lạ. "
            "Theo một bác sĩ, uong nuoc chanh nong se tieu diet virus corona trong co the!!! "
            "Hãy chia sẻ cho người thân. Cảm ơn mọi người đã đọc đến đây.")
    top = kb.search(post, top_k=3)
    print(f"    top match: {top[0].claim_id} score={top[0].score}")
    assert top[0].claim_id == "C1"

    exact = kb.best_match("Uống nước chanh nóng có thể tiêu diệt virus corona trong cơ thể")
    assert exact is not None and exact.is_debunked and exact.score > 0.9

    # Time cutoff: a post from 2019 cannot be matched with a 2020 debunk (anti-leakage)
    assert all(m.claim_id != "C1" for m in kb.search(post, top_k=3, before="2019-12-31"))
    # dd/mm/yyyy is read day-first: C2 published 15 June 2024 -> visible on 01/07/2024, not on 01/06/2024
    sea = "Sinh vật lạ khổng lồ dạt vào bờ biển miền Trung"
    assert kb.search(sea, top_k=1, before="01/07/2024")[0].claim_id == "C2"
    assert all(m.claim_id != "C2" for m in kb.search(sea, top_k=3, before="01/06/2024"))

    # Unrelated text stays below threshold
    assert kb.best_match("Đội tuyển bóng đá giành chiến thắng trong trận giao hữu tối qua") is None

    kb_dir = os.path.join(tmp, "kb")
    kb.save(kb_dir)
    kb2 = ClaimKnowledgeBase.load(kb_dir)
    assert kb2.search(post, top_k=1)[0].claim_id == "C1"
    assert kb2.threshold == kb.threshold
    print("    [OK] claim KB passed.")
    return kb


def test_sources() -> None:
    print("\n[3] Source credibility...")
    sc = SourceCredibility()
    cases = {
        "https://vnexpress.net/abc": "trusted",
        "suckhoe.vnexpress.net/xyz": "trusted",
        "https://www.moh.gov.vn/tin": "trusted",           # any *.gov.vn
        "https://nhandan.org.vn/a": "trusted",              # legit old domain, not a look-alike
        "http://vnexpres.net/tin-nong": "lookalike",        # typo-squatting
        "https://vnexpress.net.tin-nong.xyz/a": "lookalike",  # embedded trusted domain
        "https://tuoitre-news.online/b": "lookalike",
        "https://myblog.wordpress.com": "unknown",
    }
    for url, expected in cases.items():
        got = sc.check_domain(url)["status"]
        print(f"    {url:<40} -> {got}")
        assert got == expected, f"{url}: expected {expected}, got {got}"

    urls = extract_urls("Nguồn: vnexpres.net/tin-a và https://tuoitre.vn/b, xem thêm tại TP.HCM.")
    assert {"vnexpres.net/tin-a", "https://tuoitre.vn/b"} <= set(urls) and len(urls) == 2
    print("    [OK] source credibility passed.")


def test_checker(kb: ClaimKnowledgeBase) -> None:
    print("\n[4] PostChecker...")
    checker = PostChecker(kb)
    r = checker.check(CLICKBAIT.replace("diệt sạch virus", "tiêu diệt virus corona trong cơ thể"))
    print(f"    debunked post -> {r['hint']}: {r['message']}")
    assert r["hint"] == "matches_debunked" and r["evidence"][0]["source_url"]
    assert r["style_signals"] and r["clickbait_score"] > 0.5

    assert checker.check("Ngân hàng Nhà nước điều chỉnh lãi suất điều hành")["hint"] == "matches_verified"
    assert checker.check("Tin mới nhất xem tại vnexpres.net/abc")["hint"] == "suspicious_source"
    assert checker.check("Hôm nay trời đẹp, cả nhà đi dạo công viên.", model_prob_fake=0.9)["hint"] == "model_suspicious"
    assert checker.check("SỐC!!! Không thể tin nổi, chia sẻ ngay!!! 😱")["hint"] == "clickbait_style"
    assert checker.check("Hôm nay trời đẹp, cả nhà đi dạo công viên.")["hint"] == "no_evidence"
    # Checking a 2019 post: the 2020 debunk did not exist yet
    old = checker.check("Uống nước chanh nóng có thể tiêu diệt virus corona trong cơ thể", post_time="2019-01-01")
    assert old["hint"] != "matches_debunked"
    print("    [OK] PostChecker passed.")


def test_eval_script(kb: ClaimKnowledgeBase) -> None:
    print("\n[5] Evidence-layer evaluation (scripts/check_post.py)...")
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts"))
    from check_post import evaluate

    df = pd.DataFrame({
        "text": [
            "Uống nước chanh nóng có thể tiêu diệt virus corona trong cơ thể, chia sẻ gấp!!!",
            "SỐC!!! Phát hiện sinh vật lạ khổng lồ dạt vào bờ biển miền Trung",
            "Ngân hàng Nhà nước điều chỉnh lãi suất điều hành",
            "Hôm nay trời đẹp, cả nhà đi dạo công viên.",
        ],
        "label": [1, 1, 0, 0],
        "timestamp": ["2020-05-01", "2024-01-01", "2024-09-01", "2024-09-01"],
    })
    res = evaluate(kb, df, "post")
    row = next(r for r in res["claim_matching"] if r["threshold"] == 0.45)
    # Post 2 is BEFORE its debunk (2024-06-15): with the cutoff only post 1 can be matched
    assert row["coverage_of_fake"] == 0.5 and row["precision_fake"] == 1.0
    no_cut = evaluate(kb, df, "none")
    assert next(r for r in no_cut["claim_matching"] if r["threshold"] == 0.45)["coverage_of_fake"] == 1.0
    assert res["clickbait_auc"] is not None
    print("    [OK] evaluation passed (time cutoff changes coverage 50% -> 100%, as expected).")


def test_convert_reintel(tmp: str) -> None:
    print("\n[6] ReINTEL -> VMTF schema converter (scripts/convert_reintel.py)...")
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts"))
    from convert_reintel import convert, parse_count, parse_timestamp
    from validate_and_preprocess_dataset import STANDARD_COLUMNS

    cases = {"12": 12, "12.0": 12, "1,2K": 1200, "3.4k": 3400, "2 tr": 2_000_000, "1.234": 1234,
             "12,345,678": 12_345_678, "": None, "unknown": None, 7: 7, "-3": None}
    for raw, expected in cases.items():
        got = parse_count(raw)
        assert (expected is None and got != got) or got == expected, f"parse_count({raw!r}) = {got}, expected {expected}"
    assert parse_timestamp("1584000000") == "2020-03-12 08:00:00"
    assert parse_timestamp("15/03/2020 10:00") == "2020-03-15 10:00:00"
    assert parse_timestamp("not a date") == ""

    img_dir = os.path.join(tmp, "reintel_images")
    os.makedirs(os.path.join(img_dir, "sub"), exist_ok=True)
    for name in ("101_0.jpg", "101_1.jpg", os.path.join("sub", "abc123.png")):
        open(os.path.join(img_dir, name), "wb").close()

    raw = pd.DataFrame({
        "id": ["101", "102", "103", "104"],
        "user_name": ["u1", "u2", "u1", "u3"],
        "post_message": ["Tin một", "Tin hai", "Tin ba", "Tin bốn không nhãn"],
        "timestamp_post": ["1584000000", "1584003600", "", "1584007200"],
        "num_like_post": ["1,2K", "12.0", "", "5"],
        "num_comment_post": ["3", "", "1", "0"],
        "num_share_post": ["0", "7", "2", "1"],
        "image": ["", "['https://cdn.example.com/x/abc123.png?sig=1']", "https://cdn.example.com/none.jpg", ""],
        "label": ["1", "0", "0", ""],
    })
    out = convert(raw, image_dir=img_dir)
    assert list(out.columns) == STANDARD_COLUMNS
    assert len(out) == 3, "Unlabeled row must be dropped by default"
    r = out.set_index("sample_id")
    assert r.loc["REINTEL_101", "image_1_local_path"] == "101_0.jpg"       # matched by post id
    assert r.loc["REINTEL_101", "image_2_local_path"] == "101_1.jpg"
    assert r.loc["REINTEL_102", "image_1_local_path"] == "sub/abc123.png"   # matched by URL basename
    assert r.loc["REINTEL_102", "image_1_url"].startswith("https://")
    assert r.loc["REINTEL_103", "image_1_local_path"] == ""                 # URL without local file
    assert r.loc["REINTEL_101", "like_count"] == 1200 and r.loc["REINTEL_102", "like_count"] == 12
    assert r.loc["REINTEL_101", "label"] == 1 and r.loc["REINTEL_101", "raw_label"] == "unreliable"
    assert r.loc["REINTEL_101", "source"] == "u1" and r.loc["REINTEL_103", "timestamp"] == ""
    assert len(convert(raw, image_dir=img_dir, keep_unlabeled=True)) == 4
    print(f"    report: {out.attrs['report']}")
    print("    [OK] converter passed.")


if __name__ == "__main__":
    print("=" * 60)
    print("RUNNING VMTF-NET STEP 5 CHECKS (style features + evidence layer, offline)")
    print("=" * 60)
    with tempfile.TemporaryDirectory() as tmp:
        test_style()
        kb = test_claim_kb(tmp)
        test_sources()
        test_checker(kb)
        test_eval_script(kb)
        test_convert_reintel(tmp)
    print("\n" + "=" * 60)
    print("[OK] ALL STEP 5 CHECKS PASSED SUCCESSFULLY!")
    print("=" * 60)
