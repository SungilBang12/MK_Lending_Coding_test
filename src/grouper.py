"""[D] 스무딩·그룹핑 (최소 버전: 물리 세그먼트만. 논리 문서 복원은 후속 단계)."""

from __future__ import annotations

from src.evaluator import segments_of
from src.extractor import PageRecord


def smooth_labels(
    pages: list[PageRecord], labels: list[str], confidences: list[float]
) -> tuple[list[str], list[dict]]:
    """최소 버전: 보정 없이 통과."""
    return labels, []


def group_documents(pages: list[PageRecord], labels: list[str]) -> dict:
    physical = [
        {"label": lab, "start_page": s, "end_page": e} for lab, s, e in segments_of(labels)
    ]
    return {"physical_segments": physical, "logical_documents": []}
