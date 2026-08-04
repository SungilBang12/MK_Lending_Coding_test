"""[B] 룰 기반 1차 분류: 배타적 시그니처(문서 헤더·폼 번호) 앵커 사전 방식.

- 상단 30%(head) 매칭은 가중치 100%, 본문 매칭은 60% (URLA처럼 시그니처가
  푸터에 있는 양식이 있어 본문 검색도 유지한다).
- 점수 1위가 임계값 이상이고 2위와의 격차가 충분할 때만 확정한다.
  미달·복수 유형 근접 매칭이면 LLM으로 위임(needs_llm=True).
- "Realtor", "income" 같은 일반 단어는 앵커로 쓰지 않는다(URLA 본문의 고용주명
  "Veronica Salazar Realtor" 오분류 함정 — 실측).
"""

from __future__ import annotations

from dataclasses import dataclass

from src.extractor import PageRecord

LABELS = ["URLA_1003", "INCOME_DOC", "CREDIT_REPORT", "TITLE_REPORT", "OTHER"]

# 단일 앵커: (문구, 가중치). 모두 소문자 비교.
ANCHORS: dict[str, list[tuple[str, int]]] = {
    "URLA_1003": [
        ("uniform residential loan application", 5),
        ("freddie mac form 65", 5),
        ("fannie mae form 1003", 5),
    ],
    "CREDIT_REPORT": [
        ("370 reed rd", 5),          # Xactus 본사 주소 (tri-merge 헤더·소비자 안내문 공통)
        ("xactus", 4),
        ("credit score disclosure", 5),
        ("your credit score and the price you pay for credit", 5),  # 대출자 고지서
        ("notice to the home loan applicant", 4),
        ("tradeline", 2),
        ("fico", 2),
        ("repositories", 2),
    ],
    "TITLE_REPORT": [
        ("preliminary report", 5),
        ("fidelity national title", 5),
        ("clta preliminary report form", 5),
        ("commitment for title insurance", 5),
        ("american land title association", 4),
        ("title commitment", 5),
        ("settlement services", 3),
    ],
    "INCOME_DOC": [
        ("profit & loss statement", 5),
        ("profit and loss statement", 5),
        ("earnings statement", 5),
        ("wage and income transcript", 5),
        ("this product contains sensitive taxpayer data", 4),  # IRS 트랜스크립트 헤더
        ("form w-2", 4),
        ("form 1040", 4),
        ("wage and tax statement", 4),
        ("employer identification number (ein)", 3),
        ("gross pay", 3),
    ],
}

# 결합 앵커: 모든 문구가 동시에 나타날 때만 점수 부여 (개별로는 일반 단어라 위험한 것들)
COMBO_ANCHORS: dict[str, list[tuple[tuple[str, ...], int]]] = {
    "CREDIT_REPORT": [(("transunion", "experian", "equifax"), 4)],
    "TITLE_REPORT": [(("exhibit a", "legal description"), 5), (("schedule a", "schedule b"), 3)],
}

BODY_DISCOUNT = 0.6
SCORE_THRESHOLD = 5.0  # 배타적 시그니처 1개가 head에서 잡히는 수준
MARGIN_THRESHOLD = 3.0


@dataclass
class RuleResult:
    label: str  # 확정 라벨 (미확정 시 최고 점수 라벨, needs_llm=True)
    confidence: float
    evidence: str = ""
    needs_llm: bool = False


def _score_label(label: str, head: str, body: str) -> tuple[float, list[str]]:
    score, hits = 0.0, []
    for phrase, w in ANCHORS.get(label, []):
        if phrase in head:
            score += w
            hits.append(phrase)
        elif phrase in body:
            score += w * BODY_DISCOUNT
            hits.append(phrase)
    for phrases, w in COMBO_ANCHORS.get(label, []):
        if all(p in body for p in phrases):
            score += w
            hits.append("+".join(phrases))
    return score, hits


def classify_page(rec: PageRecord) -> RuleResult:
    head = rec.head_text.lower()
    body = rec.text.lower()
    scores: dict[str, float] = {}
    evidences: dict[str, list[str]] = {}
    for label in ANCHORS:
        s, hits = _score_label(label, head, body)
        scores[label] = round(s, 2)
        evidences[label] = hits

    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    top_label, top = ranked[0]
    second = ranked[1][1]

    if top >= SCORE_THRESHOLD and (top - second) >= MARGIN_THRESHOLD:
        # 점수·격차가 클수록 신뢰 상승, 상한 0.98
        confidence = min(0.98, 0.80 + 0.01 * top + 0.01 * (top - second))
        return RuleResult(
            label=top_label,
            confidence=round(confidence, 2),
            evidence="; ".join(evidences[top_label][:4]),
            needs_llm=False,
        )

    # 미확정: OTHER 기본값이 아니라 LLM 위임 (LLM 불가 시 호출측에서 OTHER 처리)
    return RuleResult(
        label=top_label if top > 0 else "OTHER",
        confidence=round(min(0.5, top / (SCORE_THRESHOLD * 2)), 2),
        evidence="; ".join(evidences.get(top_label, [])[:4]),
        needs_llm=True,
    )
