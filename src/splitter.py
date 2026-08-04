"""[E] 문서별 PDF 분리: 논리 문서 복원 결과대로 원본 페이지를 재조립해
문서 단위 PDF를 생성한다 (셔플 해제된 순서, 정방향으로 회전 보정).

셔플 패키지는 정방향 콘텐츠 위에 /Rotate 메타데이터만 씌워 회전시킨 것이므로
(원본 정답 PDF와 텍스트 완전 일치 + 원본은 전부 rot=0으로 실측 확인),
복사한 페이지의 회전을 0으로 리셋하면 정방향이 복원된다.

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
            dst[-1].set_rotation(0)  # 셔플 시 씌워진 /Rotate 제거 → 정방향
        name = d["doc_id"].replace("#", "_") + ".pdf"
        dst.save(out / name)
        dst.close()
        written.append(name)
    src.close()
    return written
