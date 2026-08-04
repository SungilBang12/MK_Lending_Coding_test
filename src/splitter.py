"""[E] 문서별 PDF 분리: 논리 문서 복원 결과대로 원본 페이지를 재조립해
문서 단위 PDF를 생성한다 (셔플 해제된 순서, /Rotate 보존).

출력: output/<pkg>/documents/<LABEL>_<n>.pdf
주의: 생성물은 과제 데이터이므로 저장소에 커밋하지 않는다 (.gitignore).
"""

from __future__ import annotations

from pathlib import Path

import fitz


def write_document_pdfs(input_pdf: str, documents: dict, out_dir: str | Path) -> list[str]:
    out = Path(out_dir) / "documents"
    out.mkdir(parents=True, exist_ok=True)
    src = fitz.open(input_pdf)
    written = []
    for d in documents["logical_documents"]:
        dst = fitz.open()
        for page_no in d["reconstructed_order"]:
            dst.insert_pdf(src, from_page=page_no - 1, to_page=page_no - 1)
        name = d["doc_id"].replace("#", "_") + ".pdf"
        dst.save(out / name)
        dst.close()
        written.append(name)
    src.close()
    return written
