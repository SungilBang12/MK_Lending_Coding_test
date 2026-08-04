"""LLM 케스케이드 로직을 mock provider로 검증한다 (API 키 불필요)."""

import pytest

from src.extractor import PageRecord
from src.llm_classifier import LLMCascade, LLMProvider, build_user_prompt, parse_llm_json


class MockProvider(LLMProvider):
    def __init__(self, text_responses=None, vision_responses=None):
        self.text_responses = list(text_responses or [])
        self.vision_responses = list(vision_responses or [])
        self.text_calls = 0
        self.vision_calls = 0

    def complete_text(self, system, user):
        self.text_calls += 1
        return self.text_responses.pop(0), {"input_tokens": 100, "output_tokens": 20}

    def complete_vision(self, system, user, png):
        self.vision_calls += 1
        return self.vision_responses.pop(0), {"input_tokens": 1500, "output_tokens": 20}


def make_page(text="some text " * 30, page_no=2, needs_vision=False):
    return PageRecord(
        page_no=page_no, text=text, head_text=text[:100], rotation=0,
        n_chars=len(text), n_images=0, needs_vision=needs_vision,
    )


PAGES = [make_page(page_no=1), make_page(page_no=2), make_page(page_no=3)]
LABELS = ["CREDIT_REPORT", "OTHER", "CREDIT_REPORT"]
GOOD = '{"label": "INCOME_DOC", "confidence": 0.9, "evidence": "P&L", "internal_page": null}'


def test_parse_llm_json_extracts_object_from_noise():
    res = parse_llm_json("여기 결과입니다:\n" + GOOD + "\n감사합니다")
    assert res.label == "INCOME_DOC"
    assert res.confidence == 0.9


def test_parse_llm_json_rejects_bad_label():
    with pytest.raises(Exception):
        parse_llm_json('{"label": "UNKNOWN", "confidence": 0.5, "evidence": ""}')


def test_text_path_success_skips_vision():
    provider = MockProvider(text_responses=[GOOD])
    cascade = LLMCascade(pdf_path="unused.pdf", provider=provider)
    res = cascade.classify(PAGES[1], PAGES, LABELS)
    assert res.label == "INCOME_DOC"
    assert res.method == "llm_text"
    assert provider.vision_calls == 0


def test_parse_failure_retries_once():
    provider = MockProvider(text_responses=["말도 안 되는 응답", GOOD])
    cascade = LLMCascade(pdf_path="unused.pdf", provider=provider)
    res = cascade.classify(PAGES[1], PAGES, LABELS)
    assert res.label == "INCOME_DOC"
    assert provider.text_calls == 2


def test_prompt_contains_neighbor_context():
    prompt = build_user_prompt(PAGES[1], PAGES, LABELS)
    assert "직전 페이지 1" in prompt
    assert "CREDIT_REPORT" in prompt
    assert "직후 페이지 3" in prompt
    assert "2,000자" in prompt
