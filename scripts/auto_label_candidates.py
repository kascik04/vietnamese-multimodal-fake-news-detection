#!/usr/bin/env python3
"""
Weak-supervision / heuristic auto-labeling for data/vmtf_candidates.py (55-column VMTF schema).
Uses only the Python standard library so it works in any environment (local or Colab).

Signals combined:
1. Source credibility: verified state TV, mainstream press, and hospitals/medical centers (-3)
   vs. folk-remedy, showbiz-gossip, superstition, and sensational tabloid channels (+3).
2. Editorial vs. sensational framing in text: official warnings/debunking/law-enforcement (-1 per match)
   vs. miracle cures, unverified home remedies, sensational rumors, phone solicitation (+1/+2).
3. Search query context from collector notes: mainstream news/weather/health queries (-1)
   vs. rumor/hoax/clickbait queries (+1).
"""

import argparse
import collections
import csv
import hashlib
import os
import random
import re
import sys
from typing import Dict, List, Tuple

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

TRUSTED_SOURCES = [
    "vtv", "chuyển động 24h", "ấn tượng vtv", "vtc", "antv", "công an nhân dân",
    "nhân dân", "quốc hội", "qpvn", "vov", "thông tấn xã", "ttxvn", "hanoitv",
    "htv", "thvl", "vinh long", "đồng tháp", "thđt", "nghệ an tv", "bptv",
    "anninhthudo", "an ninh thủ đô", "đài phát thanh", "truyền hình",
    "sức khỏe & đời sống", "sức khỏe và đời sống", "thanh niên", "tuổi trẻ",
    "tiền phong", "viettimes", "lao động", "dân trí", "vnexpress", "vietnamnet",
    "pháp luật", "người lao động", "chính phủ", "alobacsi", "bệnh viện",
    "bộ y tế", "vnvc", "tâm anh", "medlatec", "vinmec", "đại học y",
]

FAKE_SOURCES = [
    "đi soi sao", "đông dược", "gia truyền", "lương y", "thần y", "thực dưỡng",
    "chân thiện mỹ", "năng lượng tái sinh", "năng lượng gốc", "chữa lành",
    "người nổi tiếng", "chuyện nghệ sĩ", "xóm hóng", "69smedia", "đầu xanh mỏ hỗn",
    "nghiện hay chuyện", "hóng biến", "sao việt", "bí mật", "tâm linh", "phong thủy",
    "tử vi", "thầy bói", "cây thuốc", "bài thuốc",
]

REAL_TEXT_PATTERNS = [
    "cảnh báo lừa đảo", "công an cảnh báo", "bác bỏ tin đồn", "thông tin sai sự thật",
    "xử phạt", "khởi tố", "bắt tạm giam", "triệt phá", "cục an toàn thực phẩm",
    "bộ y tế", "bộ công an", "đính chính", "cảnh giác", "cơ quan chức năng",
    "bản tin", "thời sự", "khuyến cáo", "cảnh báo thủ đoạn",
]

FAKE_TEXT_PATTERNS = [
    "trị dứt điểm", "chữa dứt điểm", "chữa bách bệnh", "trị bách bệnh",
    "chữa khỏi hoàn toàn", "bài thuốc dân gian", "thần dược", "nước kiềm",
    "thực dưỡng", "không cần thuốc", "gây đột quỵ", "bằng tâm linh",
    "trục vong", "bắt ma", "trừ tà", "bùa ngải", "ngày tận thế", "cúng giải hạn",
    "lộ clip nóng", "qua đời đột ngột", "bóc phốt", "phong sát", "ngoại tình",
    "vỡ nợ tiền tỷ", "mổ nội tạng", "việc nhẹ lương cao", "cam kết lãi",
    "sàn giao dịch ảo", "app kiếm tiền", "chấn động", "kinh hoàng", "rúng động",
    "tại nhà", "dân gian", "mẹo chữa", "khỏi hẳn", "xôn xao", "thực hư",
]

RUMOR_QUERIES = {
    "drama showbiz việt", "sao kê từ thiện nghệ sĩ", "nghệ sĩ qua đời đột ngột",
    "bóc phốt nghệ sĩ", "thuốc gia truyền trị dứt điểm", "lộ clip nóng showbiz",
    "thực hư nghệ sĩ qua đời", "bắt cóc trẻ em mổ nội tạng", "chữa bệnh qua mạng xã hội",
    "scandal ca sĩ diễn viên", "thần y trị bách bệnh", "bài thuốc dân gian chữa ung thư",
    "uống nước kiềm chữa ung thư", "chữa bệnh bằng tâm linh", "phong sát nghệ sĩ",
    "tin đồn nghệ sĩ", "thực dưỡng chữa bách bệnh", "vắc xin gây đột quỵ",
    "thực phẩm chức năng trị bách bệnh", "nghệ sĩ vỡ nợ tiền tỷ",
    "tin đồn ly hôn ngoại tình showbiz", "đa cấp biến tướng", "tín dụng đen bốc bát họ",
    "đầu tư tiền ảo cam kết lãi", "phương thuốc", "thuốc chữa bệnh", "thực phẩm chức năng",
    "sự thật vụ án chấn động", "tin đồn vỡ đập", "tin đồn thất thiệt hoang mang",
}

REAL_QUERIES = {
    "mưa lớn", "ngập lụt", "lũ quét", "lũ lụt", "sạt lở", "bão", "hạn hán",
    "động đất", "thiên tai", "giao thông", "sức khỏe", "bệnh viện", "y tế",
    "điều trị", "cúm", "covid", "vaccine", "vắc xin", "dịch bệnh",
    "xử phạt tung tin giả", "ngộ độc thực phẩm chức năng giả",
}


def score_row(row: Dict[str, str]) -> Tuple[int, float]:
    src = (row.get("source") or "").lower()
    txt_raw = row.get("text") or ""
    txt = txt_raw.lower()
    notes = (row.get("notes") or "").lower()
    m = re.search(r"query=([^;]+)", notes)
    q = m.group(1).strip() if m else ""

    score = 0
    if any(k in src for k in TRUSTED_SOURCES):
        score -= 3
    if any(k in src for k in FAKE_SOURCES):
        score += 3
    for k in REAL_TEXT_PATTERNS:
        if k in txt:
            score -= 1
    for k in FAKE_TEXT_PATTERNS:
        if k in txt:
            score += 1
    if re.search(r"\b0[35789]\d{8}\b", txt):
        score += 2
    if "!" in txt_raw or "?" in txt_raw:
        score += 1
    if q in RUMOR_QUERIES:
        score += 1
    if q in REAL_QUERIES:
        score -= 1

    label = 1 if score > 0 else 0
    conf = min(0.95, 0.60 + 0.07 * abs(score))
    return label, round(conf, 2)


def fill_ratios(row: Dict[str, str]) -> None:
    def _num(val: str) -> float:
        try:
            return max(0.0, float(val))
        except (ValueError, TypeError):
            return 0.0

    like = _num(row.get("like_count", ""))
    comment = _num(row.get("comment_count", ""))
    share = _num(row.get("share_count", ""))
    row["like_count"] = str(int(like))
    row["comment_count"] = str(int(comment))
    row["share_count"] = str(int(share))
    if not str(row.get("angry_count", "")).strip():
        row["angry_count"] = "0"
    if not str(row.get("haha_count", "")).strip():
        row["haha_count"] = "0"
    if not str(row.get("like_share_ratio", "")).strip():
        row["like_share_ratio"] = f"{like / (share + 1.0):.4f}"
    if not str(row.get("comment_share_ratio", "")).strip():
        row["comment_share_ratio"] = f"{comment / (share + 1.0):.4f}"
    if not str(row.get("engagement_ratio", "")).strip():
        row["engagement_ratio"] = f"{(like + comment + share * 2.0) / (like + 1.0):.4f}"


def group_key(row: Dict[str, str]) -> str:
    sha = (row.get("image_1_sha256") or "").strip()
    if sha:
        return f"sha:{sha}"
    txt = re.sub(r"\s+", " ", (row.get("text") or "").lower()).strip()[:120]
    return "txt:" + hashlib.md5(txt.encode("utf-8")).hexdigest()


def split_groups(rows: List[Dict[str, str]], seed: int = 42) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], List[Dict[str, str]]]:
    groups: Dict[str, List[Dict[str, str]]] = collections.defaultdict(list)
    for r in rows:
        groups[group_key(r)].append(r)

    by_label: Dict[str, List[List[Dict[str, str]]]] = {"0": [], "1": []}
    for g_rows in groups.values():
        maj = "1" if sum(1 for r in g_rows if str(r["label"]) == "1") * 2 >= len(g_rows) else "0"
        by_label[maj].append(g_rows)

    rng = random.Random(seed)
    train, val, test = [], [], []
    for lbl, g_list in by_label.items():
        rng.shuffle(g_list)
        n_total = sum(len(g) for g in g_list)
        n_train_target = int(round(n_total * 0.70))
        n_val_target = int(round(n_total * 0.15))
        c_train, c_val = 0, 0
        for g in g_list:
            if c_train < n_train_target:
                train.extend(g)
                c_train += len(g)
            elif c_val < n_val_target:
                val.extend(g)
                c_val += len(g)
            else:
                test.extend(g)
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


def main() -> None:
    parser = argparse.ArgumentParser(description="Auto-label vmtf_candidates.csv and split into train/val/test")
    parser.add_argument("--input", default="data/vmtf_candidates.csv")
    parser.add_argument("--output", default="data/vmtf_candidates.csv")
    parser.add_argument("--split_dir", default="data", help="If set, also write train.csv, val.csv, test.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    counts = collections.Counter()
    for r in rows:
        cur = str(r.get("label", "")).strip()
        if cur in ("0", "1"):
            lbl = int(cur)
        else:
            lbl, conf = score_row(r)
            r["label"] = str(lbl)
            r["raw_label"] = "fake" if lbl == 1 else "real"
            r["label_source"] = "weak_supervision_v1"
            r["label_confidence"] = str(conf)
            r["annotation_status"] = "auto_labeled"
        fill_ratios(r)
        counts[int(r["label"])] += 1

    with open(args.output, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[OK] Labeled {len(rows)} rows in {args.output} -> label 0 (real): {counts[0]}, label 1 (fake): {counts[1]}")

    if args.split_dir:
        os.makedirs(args.split_dir, exist_ok=True)
        train, val, test = split_groups(rows, seed=args.seed)
        for name, split_rows in (("train", train), ("val", val), ("test", test)):
            out_path = os.path.join(args.split_dir, f"{name}.csv")
            with open(out_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(split_rows)
            c = collections.Counter(int(r["label"]) for r in split_rows)
            print(f"  -> {out_path}: {len(split_rows)} rows (0: {c[0]}, 1: {c[1]})")


if __name__ == "__main__":
    main()
