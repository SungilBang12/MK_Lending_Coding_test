"""[E] 리포트 (최소 버전: 자리표시 HTML. 시각화는 후속 단계)."""

from __future__ import annotations

from pathlib import Path

from src.extractor import PageRecord


def write_html_report(
    out_dir: Path,
    pages: list[PageRecord],
    labels: list[str],
    confidences: list[float],
    methods: list[str],
    documents: dict,
) -> None:
    html = "<html><body><h1>report placeholder</h1></body></html>"
    (Path(out_dir) / "report.html").write_text(html)
