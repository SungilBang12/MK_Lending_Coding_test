"""[A] 페이지 추출: PyMuPDF로 페이지별 텍스트·회전각·이미지 수·내부 페이지 번호를 뽑는다.

- get_text()는 /Rotate를 자동 보정하므로 텍스트 경로에선 회전을 무시할 수 있다.
- 블록 bbox는 회전이 반영된 시각 좌표계로 반환됨을 실측으로 확인 → 상단 30% 필터를
  좌표 그대로 적용한다.
- 텍스트가 VISION_MIN_CHARS 미만인 페이지는 Vision 폴백 후보로 표시한다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

import fitz  # PyMuPDF

VISION_MIN_CHARS = 200
HEAD_RATIO = 0.30  # 페이지 상단 30%

# "Page 7 of 11" / "Page: 7 of 11" / "PAGE 2 OF 2" / "Page 7/11" (대소문자 무시)
_PAGE_X_OF_Y = re.compile(r"page[:\s]*(\d{1,3})\s*(?:of|/)\s*(\d{1,3})", re.IGNORECASE)
# 접두어 없는 "7 of 11" (URLA가 이 형식)
_X_OF_Y = re.compile(r"\b(\d{1,3})\s+of\s+(\d{1,3})\b", re.IGNORECASE)
# "of M" 없는 "Page 2" (Title Report CLTA 양식)
_PAGE_ONLY = re.compile(r"\bPage\s+(\d{1,3})\b")


@dataclass
class PageRecord:
    page_no: int  # 1-based, 셔플된 PDF 내 물리적 위치
    text: str
    head_text: str  # 상단 30% 영역 텍스트
    rotation: int
    n_chars: int
    n_images: int
    internal_page: Optional[int] = None  # 문서 내부 페이지 번호 N
    internal_total: Optional[int] = None  # "N of M"의 M (없으면 None)
    needs_vision: bool = False


def parse_internal_page(text: str) -> tuple[Optional[int], Optional[int]]:
    """텍스트에서 내부 페이지 번호 (N, M)을 추출. 우선순위: 'Page N of M' > 'N of M' > 'Page N'."""
    m = _PAGE_X_OF_Y.search(text)
    if m:
        return int(m.group(1)), int(m.group(2))
    m = _X_OF_Y.search(text)
    if m:
        n, total = int(m.group(1)), int(m.group(2))
        if n <= total:  # "2021 of ..." 같은 오탐 방지
            return n, total
    m = _PAGE_ONLY.search(text)
    if m:
        return int(m.group(1)), None
    return None, None


def _head_text(page: fitz.Page) -> str:
    """회전 반영 좌표 기준 상단 30% 영역의 텍스트."""
    cutoff = page.rect.height * HEAD_RATIO
    blocks = page.get_text("blocks")
    parts = [b[4] for b in blocks if b[1] < cutoff]  # b[1] = y0
    return "\n".join(parts)


def extract_pages(pdf_path: str) -> list[PageRecord]:
    doc = fitz.open(pdf_path)
    records = []
    for i, page in enumerate(doc):
        text = page.get_text()
        n, total = parse_internal_page(text)
        rec = PageRecord(
            page_no=i + 1,
            text=text,
            head_text=_head_text(page),
            rotation=page.rotation,
            n_chars=len(text),
            n_images=len(page.get_images()),
            internal_page=n,
            internal_total=total,
            needs_vision=len(text.strip()) < VISION_MIN_CHARS,
        )
        records.append(rec)
    doc.close()
    return records


def render_page_png(pdf_path: str, page_no: int, dpi: int = 150) -> bytes:
    """Vision 폴백용 페이지 렌더링. get_pixmap은 /Rotate를 적용해 정방향 이미지를 만든다."""
    doc = fitz.open(pdf_path)
    page = doc[page_no - 1]
    pix = page.get_pixmap(dpi=dpi)
    data = pix.tobytes("png")
    doc.close()
    return data
