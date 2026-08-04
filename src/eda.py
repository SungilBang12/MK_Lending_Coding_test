"""EDA 스크립트: 패키지 PDF의 페이지별 통계를 표로 출력한다.

사용: python -m src.eda data/testing/01.990145627_fixed.pdf
"""

import sys

from src.extractor import extract_pages


def main(pdf_path: str) -> None:
    records = extract_pages(pdf_path)
    print(f"{pdf_path}: {len(records)} pages")
    print(f"{'pg':>3} {'rot':>3} {'chars':>6} {'imgs':>4} {'internal':>9} {'vision':>6}  head")
    for r in records:
        internal = f"{r.internal_page}/{r.internal_total}" if r.internal_page else "-"
        head = " ".join(r.head_text.split())[:60]
        print(
            f"{r.page_no:>3} {r.rotation:>3} {r.n_chars:>6} {r.n_images:>4} "
            f"{internal:>9} {'V' if r.needs_vision else '':>6}  {head}"
        )
    n_vision = sum(r.needs_vision for r in records)
    print(f"\nrotations: {sorted({r.rotation for r in records})}, vision candidates: {n_vision}")


if __name__ == "__main__":
    main(sys.argv[1])
