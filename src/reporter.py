"""[E] HTML 리포트: 페이지 타임라인 색 스트립, 라벨 분포, confidence 분포,
저신뢰 페이지 목록. matplotlib으로 PNG 생성 후 단일 HTML에 base64 임베드한다."""

from __future__ import annotations

import base64
import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 한글 라벨 렌더링: 설치된 한글 폰트를 우선순위로 지정 (없으면 기본 폰트로 동작)
plt.rcParams["font.family"] = [
    "Apple SD Gothic Neo", "AppleGothic", "NanumGothic", "Malgun Gothic",
    "Noto Sans CJK KR", "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False
from matplotlib.patches import Patch

from src.extractor import PageRecord

# 검증된 카테고리 팔레트 (light, 라벨별 고정 배정 — 순환 금지)
LABEL_COLORS = {
    "URLA_1003": "#2a78d6",
    "INCOME_DOC": "#eb6834",
    "CREDIT_REPORT": "#1baf7a",
    "TITLE_REPORT": "#eda100",
    "OTHER": "#e87ba4",
}
SURFACE = "#fcfcfb"
INK = "#333330"
INK_MUTED = "#6f6e66"


def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return base64.standard_b64encode(buf.getvalue()).decode()


def _style_axes(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#d8d7cf")
    ax.tick_params(colors=INK_MUTED, labelsize=9)


def _timeline_strip(labels: list[str], methods: list[str]) -> str:
    n = len(labels)
    fig, ax = plt.subplots(figsize=(max(8, n * 0.28), 1.6), facecolor=SURFACE)
    for i, lab in enumerate(labels):
        ax.bar(i + 1, 1, width=0.92, color=LABEL_COLORS[lab], edgecolor=SURFACE, linewidth=1)
        if methods[i] != "rule":  # LLM/미해결 페이지 표시
            ax.text(i + 1, 0.5, "•", ha="center", va="center", color="white", fontsize=11)
    ax.set_xlim(0.4, n + 0.6)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xticks(range(1, n + 1, 2))
    _style_axes(ax)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("페이지 (셔플된 물리 순서, • = 룰 미확정 페이지)", color=INK_MUTED, fontsize=9)
    handles = [Patch(color=c, label=l) for l, c in LABEL_COLORS.items() if l in labels]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.55),
              ncol=len(handles), frameon=False, fontsize=9, labelcolor=INK)
    return _fig_to_b64(fig)


def _label_distribution(labels: list[str]) -> str:
    order = [l for l in LABEL_COLORS if l in labels]
    counts = [labels.count(l) for l in order]
    fig, ax = plt.subplots(figsize=(6, 2.8), facecolor=SURFACE)
    bars = ax.barh(order[::-1], counts[::-1],
                   color=[LABEL_COLORS[l] for l in order[::-1]], height=0.55)
    for b, c in zip(bars, counts[::-1]):
        ax.text(b.get_width() + 0.3, b.get_y() + b.get_height() / 2, str(c),
                va="center", color=INK, fontsize=10)
    ax.set_xlim(0, max(counts) * 1.15)
    _style_axes(ax)
    ax.tick_params(axis="y", labelsize=10, labelcolor=INK)
    ax.set_xlabel("페이지 수", color=INK_MUTED, fontsize=9)
    return _fig_to_b64(fig)


def _confidence_hist(confidences: list[float]) -> str:
    fig, ax = plt.subplots(figsize=(6, 2.8), facecolor=SURFACE)
    ax.hist(confidences, bins=[i / 10 for i in range(11)], color="#2a78d6",
            edgecolor=SURFACE, linewidth=1)
    ax.axvline(0.6, color="#e34948", linewidth=1.2, linestyle="--")
    ax.text(0.59, ax.get_ylim()[1] * 0.9, "저신뢰 경계 0.6", ha="right",
            color="#e34948", fontsize=8)
    _style_axes(ax)
    ax.set_xlabel("confidence", color=INK_MUTED, fontsize=9)
    ax.set_ylabel("페이지 수", color=INK_MUTED, fontsize=9)
    return _fig_to_b64(fig)


def write_html_report(
    out_dir: Path,
    pages: list[PageRecord],
    labels: list[str],
    confidences: list[float],
    methods: list[str],
    documents: dict,
) -> None:
    out_dir = Path(out_dir)
    strip = _timeline_strip(labels, methods)
    dist = _label_distribution(labels)
    conf = _confidence_hist(confidences)

    low_conf = [
        (p.page_no, lab, c, m)
        for p, lab, c, m in zip(pages, labels, confidences, methods)
        if c < 0.6
    ]
    low_rows = "".join(
        f"<tr><td>{pn}</td><td>{lab}</td><td>{c:.2f}</td><td>{m}</td></tr>"
        for pn, lab, c, m in low_conf
    ) or "<tr><td colspan='4'>없음</td></tr>"

    page_rows = "".join(
        f"<tr><td>{p.page_no}</td>"
        f"<td><span class='chip' style='background:{LABEL_COLORS[lab]}'></span>{lab}</td>"
        f"<td>{c:.2f}</td><td>{m}</td><td>{p.rotation}°</td><td>{p.n_chars}</td>"
        f"<td>{f'{p.internal_page} of {p.internal_total}' if p.internal_page and p.internal_total else (f'page {p.internal_page}' if p.internal_page else '')}</td></tr>"
        for p, lab, c, m in zip(pages, labels, confidences, methods)
    )

    seg_rows = "".join(
        f"<tr><td>{s['label']}</td><td>{s['start_page']}</td><td>{s['end_page']}</td></tr>"
        for s in documents["physical_segments"]
    )
    doc_rows = "".join(
        f"<tr><td>{d['doc_id']}</td><td>{d['internal_total'] or '-'}</td><td>{d['n_pages']}</td>"
        f"<td>{' → '.join(map(str, d['reconstructed_order']))}</td></tr>"
        for d in documents["logical_documents"]
    )
    meta = documents.get("meta", {})

    html = f"""<!DOCTYPE html>
<html lang="ko"><head><meta charset="utf-8"><title>페이지 분류 리포트 — {meta.get('input', '')}</title>
<style>
body {{ font-family: -apple-system, 'Apple SD Gothic Neo', sans-serif; background: {SURFACE};
       color: {INK}; max-width: 1100px; margin: 2rem auto; padding: 0 1rem; }}
h1 {{ font-size: 1.4rem; }} h2 {{ font-size: 1.1rem; margin-top: 2rem; }}
img {{ max-width: 100%; }}
table {{ border-collapse: collapse; font-size: 0.85rem; width: 100%; }}
th, td {{ border-bottom: 1px solid #e4e3db; padding: 4px 10px; text-align: left; }}
th {{ color: {INK_MUTED}; font-weight: 600; }}
.chip {{ display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 6px; }}
.meta {{ color: {INK_MUTED}; font-size: 0.85rem; }}
.grid {{ display: flex; gap: 2rem; flex-wrap: wrap; }} .grid > div {{ flex: 1 1 380px; }}
</style></head><body>
<h1>대출 서류 페이지 분류 리포트</h1>
<p class="meta">입력: {meta.get('input', '?')} · {meta.get('n_pages', '?')}페이지 ·
처리 {meta.get('elapsed_seconds', '?')}초 · no_llm={meta.get('no_llm')} ·
LLM 위임 {len(meta.get('llm_delegated_pages', []))}건 · 스무딩 보정 {len(meta.get('smoothing_log', []))}건</p>

<h2>페이지 타임라인</h2><img src="data:image/png;base64,{strip}">
<div class="grid">
<div><h2>라벨 분포</h2><img src="data:image/png;base64,{dist}"></div>
<div><h2>Confidence 분포</h2><img src="data:image/png;base64,{conf}"></div>
</div>

<h2>저신뢰 페이지 (confidence &lt; 0.6)</h2>
<table><tr><th>페이지</th><th>라벨</th><th>confidence</th><th>method</th></tr>{low_rows}</table>

<h2>물리 세그먼트 ({len(documents['physical_segments'])}개)</h2>
<table><tr><th>라벨</th><th>시작</th><th>끝</th></tr>{seg_rows}</table>

<h2>논리 문서 복원 ({len(documents['logical_documents'])}개)</h2>
<table><tr><th>doc_id</th><th>내부 총 페이지(M)</th><th>페이지 수</th><th>복원 순서 (셔플된 물리 페이지 번호)</th></tr>{doc_rows}</table>

<h2>페이지별 분류 결과</h2>
<table><tr><th>페이지</th><th>라벨</th><th>conf</th><th>method</th><th>회전</th><th>문자수</th><th>내부번호</th></tr>{page_rows}</table>
</body></html>"""
    (out_dir / "report.html").write_text(html)
