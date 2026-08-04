"""[C] LLM 2차 분류: 룰이 확정하지 못한 저신뢰 페이지만 LLM에 위임한다.

- 텍스트 경로: claude-haiku-4-5-20251001 (값싸고 빠름)
- Vision 폴백: claude-sonnet-4-6 (텍스트 200자 미만이거나 텍스트 분류 실패 시,
  페이지를 150dpi PNG로 렌더링해 전송)
- LLMProvider 인터페이스로 provider 추상화 (OpenAI 등으로 교체 가능)
- 응답은 Pydantic으로 파싱·검증. 실패 시 1회 재시도 후 OTHER + confidence 0.0
- 호출 수·토큰·소요 시간을 페이지별로 로깅
"""

from __future__ import annotations

import json
import os
import re
import time
from abc import ABC, abstractmethod
from typing import Literal, Optional

from pydantic import BaseModel, Field, ValidationError

from src.extractor import PageRecord, render_page_png

DEFAULT_TEXT_MODEL = "claude-haiku-4-5-20251001"
DEFAULT_VISION_MODEL = "claude-sonnet-4-6"

LABELS_LITERAL = Literal["URLA_1003", "INCOME_DOC", "CREDIT_REPORT", "TITLE_REPORT", "OTHER"]


class PageLLMResult(BaseModel):
    label: LABELS_LITERAL
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: str = ""
    internal_page: Optional[str] = None
    method: str = "llm_text"  # 파싱 후 케스케이드가 채움


SYSTEM_PROMPT = """당신은 미국 모기지 대출 서류 패키지의 페이지 분류 전문가다.
각 페이지를 아래 5개 유형 중 하나로 분류하라.

유형 정의와 대표 시그니처:
1. URLA_1003: Uniform Residential Loan Application (Fannie Mae Form 1003 / Freddie Mac Form 65).
   시그니처: "Uniform Residential Loan Application", "Freddie Mac Form 65", "Fannie Mae Form 1003",
   Section 1~9, Lender Loan Information, Continuation Sheet, Unmarried Addendum.
2. INCOME_DOC: 소득 증빙. P&L(손익계산서), W-2, 급여명세(Earnings Statement), IRS Wage and
   Income Transcript, 1040, 1099. 시그니처: "Profit & Loss", "Form W-2", "Gross Pay",
   "This Product Contains Sensitive Taxpayer Data", 매출/비용/순이익 수치 나열.
3. CREDIT_REPORT: 신용조회 패키지. Xactus tri-merge 본체(370 Reed Rd 헤더, Repositories:
   TUC/EXP/EQX, Tradeline, FICO), 벤더 소비자 안내문, Credit Score Disclosure 등 부속 고지서 포함.
4. TITLE_REPORT: 권원 보고서/권원보험 확약서. "PRELIMINARY REPORT", "FIDELITY NATIONAL TITLE",
   "CLTA Preliminary Report Form", "Commitment for Title Insurance", Schedule A/B,
   "EXHIBIT A" + Legal Description, plat map(지적도 — 텍스트가 거의 없는 도면 페이지).
5. OTHER: 위 4개 유형에 속하지 않는 페이지.

판단 규칙:
- 문서 헤더·폼 번호 같은 배타적 시그니처를 최우선으로 본다.
- 본문에 등장하는 일반 단어(예: 고용주명 "Realtor", "income")만으로 분류하지 않는다.
- 직전/직후 페이지의 1차 분류 결과를 문맥으로 참고하라(셔플된 패키지이므로 절대적이지 않다).
- 텍스트가 거의 없는 도면/이미지 페이지는 앞뒤 문맥과 시각적 단서로 판단하라.

반드시 아래 JSON 형식으로만 답하라. 다른 텍스트를 출력하지 마라:
{"label": "URLA_1003|INCOME_DOC|CREDIT_REPORT|TITLE_REPORT|OTHER", "confidence": 0.0~1.0,
 "evidence": "판단 근거가 된 문구", "internal_page": "N of M 패턴이 있으면 기록, 없으면 null"}"""


class LLMProvider(ABC):
    """LLM provider 추상화. usage dict: {input_tokens, output_tokens}"""

    @abstractmethod
    def complete_text(self, system: str, user: str) -> tuple[str, dict]: ...

    @abstractmethod
    def complete_vision(self, system: str, user: str, png: bytes) -> tuple[str, dict]: ...


class AnthropicProvider(LLMProvider):
    def __init__(self):
        import anthropic

        self.client = anthropic.Anthropic()
        self.text_model = os.environ.get("LLM_TEXT_MODEL", DEFAULT_TEXT_MODEL)
        self.vision_model = os.environ.get("LLM_VISION_MODEL", DEFAULT_VISION_MODEL)

    def complete_text(self, system: str, user: str) -> tuple[str, dict]:
        resp = self.client.messages.create(
            model=self.text_model,
            max_tokens=512,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return self._unpack(resp)

    def complete_vision(self, system: str, user: str, png: bytes) -> tuple[str, dict]:
        import base64

        resp = self.client.messages.create(
            model=self.vision_model,
            max_tokens=512,
            system=system,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": base64.standard_b64encode(png).decode(),
                            },
                        },
                        {"type": "text", "text": user},
                    ],
                }
            ],
        )
        return self._unpack(resp)

    @staticmethod
    def _unpack(resp) -> tuple[str, dict]:
        text = next((b.text for b in resp.content if b.type == "text"), "")
        usage = {
            "input_tokens": resp.usage.input_tokens,
            "output_tokens": resp.usage.output_tokens,
        }
        return text, usage


def parse_llm_json(raw: str) -> PageLLMResult:
    """응답에서 첫 JSON 오브젝트를 추출해 Pydantic으로 검증한다."""
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        raise ValueError(f"JSON 없음: {raw[:200]!r}")
    return PageLLMResult.model_validate(json.loads(m.group(0)))


def build_user_prompt(
    page: PageRecord, pages: list[PageRecord], first_pass_labels: list[str]
) -> str:
    """대상 페이지 텍스트(2,000자 절삭) + 직전·직후 페이지의 1차 분류와 첫 200자."""
    i = page.page_no - 1
    parts = [f"[대상: 페이지 {page.page_no}] (문자 수 {page.n_chars}, 이미지 {page.n_images}개)"]
    if i > 0:
        prev = pages[i - 1]
        parts.append(
            f"[직전 페이지 {prev.page_no}] 1차 분류: {first_pass_labels[i - 1]}\n"
            f"첫 200자: {prev.text[:200]!r}"
        )
    if i < len(pages) - 1:
        nxt = pages[i + 1]
        parts.append(
            f"[직후 페이지 {nxt.page_no}] 1차 분류: {first_pass_labels[i + 1]}\n"
            f"첫 200자: {nxt.text[:200]!r}"
        )
    parts.append(f"[대상 페이지 텍스트 (최대 2,000자)]\n{page.text[:2000]}")
    return "\n\n".join(parts)


class LLMCascade:
    """텍스트 LLM → (실패·저신뢰·저텍스트 시) Vision 폴백 케스케이드."""

    VISION_CONF_THRESHOLD = 0.5  # 텍스트 LLM confidence가 이보다 낮으면 vision 재시도

    def __init__(self, pdf_path: str, provider: LLMProvider | None = None):
        self.pdf_path = pdf_path
        self.provider = provider or AnthropicProvider()
        self.log: list[dict] = []

    def classify(
        self, page: PageRecord, pages: list[PageRecord], first_pass_labels: list[str]
    ) -> PageLLMResult:
        user = build_user_prompt(page, pages, first_pass_labels)

        if not page.needs_vision:
            res = self._call(page.page_no, "llm_text", lambda: self.provider.complete_text(SYSTEM_PROMPT, user))
            if res is not None and res.confidence >= self.VISION_CONF_THRESHOLD:
                return res

        # Vision 폴백: 텍스트가 부족하거나 텍스트 분류가 실패/저신뢰
        png = render_page_png(self.pdf_path, page.page_no, dpi=150)
        res = self._call(
            page.page_no, "llm_vision", lambda: self.provider.complete_vision(SYSTEM_PROMPT, user, png)
        )
        if res is not None:
            return res
        return PageLLMResult(label="OTHER", confidence=0.0, evidence="LLM 파싱 실패", method="llm_vision")

    def _call(self, page_no: int, method: str, fn) -> PageLLMResult | None:
        """LLM 호출 + 파싱. 파싱 실패 시 1회 재시도. 실패하면 None."""
        for attempt in range(2):
            t0 = time.time()
            try:
                raw, usage = fn()
            except Exception as e:  # API 오류는 재시도 없이 기록 후 포기
                self.log.append(
                    {"page": page_no, "method": method, "error": str(e)[:200], "seconds": round(time.time() - t0, 2)}
                )
                return None
            elapsed = time.time() - t0
            self.log.append(
                {"page": page_no, "method": method, "attempt": attempt + 1,
                 "seconds": round(elapsed, 2), **usage}
            )
            try:
                res = parse_llm_json(raw)
                res.method = method
                return res
            except (ValueError, ValidationError, json.JSONDecodeError):
                continue
        return None

    def usage_summary(self) -> dict:
        ok = [e for e in self.log if "input_tokens" in e]
        return {
            "llm_calls": len(ok),
            "input_tokens": sum(e["input_tokens"] for e in ok),
            "output_tokens": sum(e["output_tokens"] for e in ok),
            "llm_seconds": round(sum(e.get("seconds", 0) for e in self.log), 2),
            "errors": [e for e in self.log if "error" in e],
            "per_page_log": self.log,
        }
