"""[F] 평가: 페이지 레벨 + 세그먼트 레벨 지표, 오답 분석 리포트.

- 페이지 레벨: accuracy, 클래스별 precision/recall/F1, macro-F1, confusion matrix
- 세그먼트 레벨: boundary F1(문서 시작점 탐지), document-level exact match
  (NER의 token-level vs entity-level 평가와 동형)
"""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass

from src.rule_classifier import LABELS


def load_gt_csv(path: str) -> list[str]:
    """page_no 오름차순 라벨 리스트를 반환한다."""
    with open(path, newline="") as f:
        rows = sorted(csv.DictReader(f), key=lambda r: int(r["page_no"]))
    return [r["label"] for r in rows]


def segments_of(labels: list[str]) -> list[tuple[str, int, int]]:
    """인접 동일 라벨 병합 → [(label, start_page, end_page)] (1-based, 양끝 포함)."""
    segs = []
    for i, lab in enumerate(labels):
        if segs and segs[-1][0] == lab and segs[-1][2] == i:
            segs[-1] = (lab, segs[-1][1], i + 1)
        else:
            segs.append((lab, i + 1, i + 1))
    return segs


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


@dataclass
class EvalReport:
    accuracy: float
    per_class: dict[str, dict[str, float]]  # label -> {precision, recall, f1, support}
    macro_f1: float
    confusion: dict[str, dict[str, int]]  # gt_label -> pred_label -> count
    boundary_precision: float
    boundary_recall: float
    boundary_f1: float
    doc_exact_match: float  # (label, start, end) 완전 일치 세그먼트 비율 (GT 기준)
    errors: list[dict]  # [{page_no, pred, gt}]


def evaluate(pred: list[str], gt: list[str]) -> EvalReport:
    assert len(pred) == len(gt), f"길이 불일치: pred={len(pred)} gt={len(gt)}"
    n = len(gt)
    correct = sum(p == g for p, g in zip(pred, gt))

    confusion: dict[str, dict[str, int]] = {g: defaultdict(int) for g in LABELS}
    for p, g in zip(pred, gt):
        confusion[g][p] += 1

    per_class = {}
    f1s = []
    for lab in LABELS:
        tp = confusion[lab][lab]
        fp = sum(confusion[g][lab] for g in LABELS if g != lab)
        fn = sum(v for k, v in confusion[lab].items() if k != lab)
        support = tp + fn
        p_, r_, f_ = _prf(tp, fp, fn)
        per_class[lab] = {"precision": p_, "recall": r_, "f1": f_, "support": support}
        if support > 0:
            f1s.append(f_)
    macro_f1 = sum(f1s) / len(f1s) if f1s else 0.0

    # 세그먼트 레벨
    pred_segs, gt_segs = segments_of(pred), segments_of(gt)
    pred_starts = {(s[1], s[0]) for s in pred_segs}  # (시작 페이지, 라벨)
    gt_starts = {(s[1], s[0]) for s in gt_segs}
    b_tp = len(pred_starts & gt_starts)
    bp, br, bf = _prf(b_tp, len(pred_starts) - b_tp, len(gt_starts) - b_tp)
    exact = len(set(pred_segs) & set(gt_segs)) / len(gt_segs) if gt_segs else 0.0

    errors = [
        {"page_no": i + 1, "pred": p, "gt": g}
        for i, (p, g) in enumerate(zip(pred, gt))
        if p != g
    ]
    return EvalReport(
        accuracy=correct / n,
        per_class=per_class,
        macro_f1=macro_f1,
        confusion={g: dict(row) for g, row in confusion.items()},
        boundary_precision=bp,
        boundary_recall=br,
        boundary_f1=bf,
        doc_exact_match=exact,
        errors=errors,
    )


def format_report(rep: EvalReport, title: str = "") -> str:
    lines = [f"## 평가 결과 {title}".rstrip(), ""]
    lines.append(f"- 페이지 accuracy: **{rep.accuracy:.4f}**")
    lines.append(f"- macro-F1: **{rep.macro_f1:.4f}**")
    lines.append(
        f"- boundary P/R/F1: {rep.boundary_precision:.3f} / "
        f"{rep.boundary_recall:.3f} / **{rep.boundary_f1:.3f}**"
    )
    lines.append(f"- document exact match: **{rep.doc_exact_match:.4f}**")
    lines.append("")
    lines.append("| label | precision | recall | f1 | support |")
    lines.append("|---|---|---|---|---|")
    for lab, m in rep.per_class.items():
        lines.append(
            f"| {lab} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} | {int(m['support'])} |"
        )
    lines.append("")
    lines.append("### Confusion matrix (행=GT, 열=pred)")
    lines.append("| GT\\pred | " + " | ".join(LABELS) + " |")
    lines.append("|---|" + "---|" * len(LABELS))
    for g in LABELS:
        row = [str(rep.confusion.get(g, {}).get(p, 0)) for p in LABELS]
        lines.append(f"| {g} | " + " | ".join(row) + " |")
    return "\n".join(lines)


def write_error_analysis(
    errors: list[dict], pages: list, out_path: str, methods: list[str] | None = None
) -> None:
    """오분류 전건에 대해 페이지 텍스트 스니펫을 포함한 error_analysis.md 생성."""
    lines = ["# 오답 분석 (error analysis)", ""]
    if not errors:
        lines.append("오분류 없음 (accuracy 1.0).")
    for e in errors:
        i = e["page_no"] - 1
        snippet = " ".join(pages[i].text.split())[:300]
        method = methods[i] if methods else "?"
        lines += [
            f"## p{e['page_no']:02d}: pred={e['pred']} / gt={e['gt']} (method={method})",
            "",
            f"- 텍스트 스니펫: `{snippet}`",
            f"- 문자 수 {pages[i].n_chars}, 이미지 {pages[i].n_images}개, 회전 {pages[i].rotation}°",
            "",
        ]
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
