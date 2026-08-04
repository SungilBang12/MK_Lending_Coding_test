"""Ground truth 생성: data/testing_answers/의 원본 문서 PDF와 셔플 패키지를
텍스트 매칭하여 페이지 레벨 정답 CSV를 만든다.

과제 안내상 "정답지 미제공"이지만, 원본 구성 문서 4종이 제공되므로
정규화 텍스트 완전 일치(폴백: difflib 유사도)로 정답을 결정적으로 복원할 수 있다.

사용: python -m src.gt_builder
출력: data/ground_truth_01.csv (page_no, label, source_doc, source_page)
"""

from __future__ import annotations

import csv
import difflib
import re
from pathlib import Path

import fitz

ANSWER_DOCS = {
    "URLA_1003": "data/testing_answers/1003 - URLA_990145627.pdf",
    "CREDIT_REPORT": "data/testing_answers/Credit Report_990145627.pdf",
    "INCOME_DOC": "data/testing_answers/INCOME - P & L_990145627.pdf",
    "TITLE_REPORT": "data/testing_answers/Title Report_990145627.pdf",
}

SHUFFLED = "data/testing/01.990145627_shuffled.pdf"
OUT_CSV = "data/ground_truth_01.csv"


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _load_pages(path: str) -> list[str]:
    doc = fitz.open(path)
    texts = [normalize(p.get_text()) for p in doc]
    doc.close()
    return texts


def build_ground_truth(shuffled_path: str = SHUFFLED, out_csv: str = OUT_CSV) -> list[dict]:
    shuffled = _load_pages(shuffled_path)
    answers = []  # (label, source_page, normalized_text)
    for label, path in ANSWER_DOCS.items():
        for j, t in enumerate(_load_pages(path)):
            answers.append((label, j + 1, t))

    rows = []
    used = set()
    for i, s_text in enumerate(shuffled):
        # 1차: 완전 일치
        match = None
        for k, (label, src_pg, a_text) in enumerate(answers):
            if k not in used and s_text == a_text:
                match = (k, label, src_pg, 1.0)
                break
        # 2차: difflib 최고 유사도
        if match is None:
            best_k, best_ratio = -1, 0.0
            for k, (label, src_pg, a_text) in enumerate(answers):
                if k in used:
                    continue
                r = difflib.SequenceMatcher(None, s_text, a_text).ratio()
                if r > best_ratio:
                    best_k, best_ratio = k, r
            label, src_pg, _ = answers[best_k]
            match = (best_k, label, src_pg, best_ratio)
        used.add(match[0])
        rows.append(
            {
                "page_no": i + 1,
                "label": match[1],
                "source_doc": Path(ANSWER_DOCS[match[1]]).name,
                "source_page": match[2],
                "match_ratio": round(match[3], 4),
            }
        )

    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    return rows


if __name__ == "__main__":
    rows = build_ground_truth()
    inexact = [r for r in rows if r["match_ratio"] < 1.0]
    from collections import Counter

    print(Counter(r["label"] for r in rows))
    print(f"{len(rows)} pages, exact={len(rows) - len(inexact)}, fuzzy={len(inexact)}")
    for r in inexact:
        print("  fuzzy:", r)
