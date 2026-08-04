"""[D] 스무딩·그룹핑.

셔플 패키지에서 "문서"의 정의는 두 가지로 갈리므로 둘 다 산출한다:
1. 물리적 세그먼트: 입력 PDF에서 인접한 동일 라벨 페이지 병합
2. 논리적 문서: 내부 페이지 번호("N of M", "Page N") + 라벨로 원본 문서 복원

주의(실측): URLA와 신용보고서가 둘 다 "N of 11" 번호를 가져 M만으로는 충돌한다.
→ 논리 문서 키는 반드시 (label, M) 결합으로 사용.
같은 (label, M)에서 동일한 N이 중복 등장하면(문서 2부 존재) 새 인스턴스로 분리한다.
내부 번호가 없는 페이지는 같은 라벨의 논리 문서에 물리 순서대로 덧붙인다.
"""

from __future__ import annotations

from src.evaluator import segments_of
from src.extractor import PageRecord


def smooth_labels(
    pages: list[PageRecord], labels: list[str], confidences: list[float]
) -> tuple[list[str], list[dict]]:
    """보수적 스무딩: 동일 라벨 시퀀스 사이에 낀 단일 이질 페이지를,
    confidence가 낮고(<0.6) 내부 페이지 번호 체계(M)가 주변 문서와 일치할 때만
    다수결로 보정한다. 셔플 패키지에선 진짜 단독 페이지일 수 있으므로 이력을 남긴다."""
    out = list(labels)
    log = []
    for i in range(1, len(labels) - 1):
        left, mid, right = out[i - 1], out[i], out[i + 1]
        if left != right or mid == left:
            continue
        if confidences[i] >= 0.6:
            continue
        # 내부 페이지 번호 연속성: 대상 페이지의 M이 이웃 페이지의 M과 일치해야 보정
        mid_total = pages[i].internal_total
        neighbor_totals = {pages[i - 1].internal_total, pages[i + 1].internal_total} - {None}
        if mid_total is None or mid_total not in neighbor_totals:
            continue
        log.append(
            {
                "page_no": pages[i].page_no,
                "from": mid,
                "to": left,
                "confidence": confidences[i],
                "reason": f"단일 이질 페이지, conf<0.6, internal_total={mid_total} 이웃과 일치",
            }
        )
        out[i] = left
    return out, log


def _logical_documents(pages: list[PageRecord], labels: list[str]) -> list[dict]:
    docs: list[dict] = []  # {label, total, pages: [(n, page_no)], unnumbered: [page_no]}

    def find_slot(label: str, total, n: int):
        """같은 (label, total)에서 번호 n이 비어 있는 첫 인스턴스. 없으면 새로 만든다."""
        for d in docs:
            if d["label"] == label and d["total"] == total and n not in {p[0] for p in d["pages"]}:
                return d
        d = {"label": label, "total": total, "pages": [], "unnumbered": []}
        docs.append(d)
        return d

    # 1) 내부 번호가 있는 페이지를 (label, M) 인스턴스에 배정 (물리 순서대로)
    for page, label in zip(pages, labels):
        if page.internal_page is not None:
            d = find_slot(label, page.internal_total, page.internal_page)
            d["pages"].append((page.internal_page, page.page_no))

    # 2) 내부 번호가 없는 페이지는 같은 라벨의 인스턴스에 물리 순서대로 덧붙임
    #    (인스턴스가 없으면 라벨 단독 문서 생성)
    for page, label in zip(pages, labels):
        if page.internal_page is None:
            target = next((d for d in docs if d["label"] == label), None)
            if target is None:
                target = {"label": label, "total": None, "pages": [], "unnumbered": []}
                docs.append(target)
            target["unnumbered"].append(page.page_no)

    result = []
    for idx, d in enumerate(docs):
        ordered = [p for _, p in sorted(d["pages"])]
        reconstructed = ordered + d["unnumbered"]
        physical = sorted(reconstructed)
        result.append(
            {
                "doc_id": f"{d['label']}#{idx + 1}",
                "label": d["label"],
                "internal_total": d["total"],
                "n_pages": len(reconstructed),
                "pages": physical,  # 셔플된 위치(물리 페이지 번호) 오름차순
                "reconstructed_order": reconstructed,  # 내부 번호순 + 무번호는 물리순 덧붙임
                "unnumbered_pages": d["unnumbered"],
            }
        )
    return result


def group_documents(pages: list[PageRecord], labels: list[str]) -> dict:
    physical = [
        {"label": lab, "start_page": s, "end_page": e} for lab, s, e in segments_of(labels)
    ]
    return {
        "physical_segments": physical,
        "logical_documents": _logical_documents(pages, labels),
    }
