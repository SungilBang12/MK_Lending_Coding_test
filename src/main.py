"""CLI 엔트리포인트.

사용:
  python -m src.main classify --input data/testing/01.990145627_shuffled.pdf \
      --output output/pkg01 [--no-llm] [--gt data/ground_truth_01.csv]
  python -m src.main build-gt
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

from src.extractor import extract_pages
from src.rule_classifier import classify_page


def run_classify(args: argparse.Namespace) -> None:
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # [A] 추출 + [B] 룰 분류
    pages = extract_pages(args.input)
    labels, confidences, methods, evidences, internals = [], [], [], [], []
    llm_targets = []
    for p in pages:
        r = classify_page(p)
        labels.append(r.label)
        confidences.append(r.confidence)
        methods.append("rule")
        evidences.append(r.evidence)
        internals.append(
            f"{p.internal_page} of {p.internal_total}"
            if p.internal_page and p.internal_total
            else (f"page {p.internal_page}" if p.internal_page else "")
        )
        if r.needs_llm:
            llm_targets.append(p.page_no)

    # [C] LLM 분류 (저신뢰 페이지만)
    cost_summary = {"llm_calls": 0, "input_tokens": 0, "output_tokens": 0, "llm_seconds": 0.0}
    if not args.no_llm and llm_targets:
        from src.llm_classifier import LLMCascade

        cascade = LLMCascade(pdf_path=args.input)
        for pno in llm_targets:
            i = pno - 1
            res = cascade.classify(pages[i], pages, labels)
            labels[i] = res.label
            confidences[i] = res.confidence
            methods[i] = res.method
            evidences[i] = res.evidence
            if res.internal_page:
                internals[i] = res.internal_page
        cost_summary = cascade.usage_summary()
    elif llm_targets:
        # --no-llm 재현 모드: 위임 대상은 OTHER 유지, method 표기만 명확히
        for pno in llm_targets:
            methods[pno - 1] = "rule_unresolved"

    # [D] 스무딩·그룹핑
    from src.grouper import group_documents, smooth_labels

    labels, smoothing_log = smooth_labels(pages, labels, confidences)
    documents = group_documents(pages, labels)

    elapsed = time.time() - t0

    # [E] 출력
    csv_path = out_dir / "page_classification.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["page_no", "label", "confidence", "method", "evidence", "internal_page"])
        for p, lab, conf, met, ev, ip in zip(
            pages, labels, confidences, methods, evidences, internals
        ):
            w.writerow([p.page_no, lab, conf, met, ev, ip])

    documents["meta"] = {
        "input": args.input,
        "n_pages": len(pages),
        "elapsed_seconds": round(elapsed, 2),
        "no_llm": bool(args.no_llm),
        "llm_delegated_pages": llm_targets,
        "smoothing_log": smoothing_log,
        "cost": cost_summary,
    }
    with open(out_dir / "documents.json", "w") as f:
        json.dump(documents, f, indent=2, ensure_ascii=False)

    from src.reporter import write_html_report

    write_html_report(out_dir, pages, labels, confidences, methods, documents)

    print(f"[E] {csv_path} / documents.json / report.html 생성 완료 ({elapsed:.1f}s)")
    print(f"    룰 확정 {len(pages) - len(llm_targets)}/{len(pages)}, LLM 위임 {len(llm_targets)}건")
    if cost_summary["llm_calls"]:
        print(f"    LLM: {cost_summary}")

    # [F] 평가
    if args.gt:
        from src.evaluator import evaluate, format_report, load_gt_csv, write_error_analysis

        gt = load_gt_csv(args.gt)
        rep = evaluate(labels, gt)
        report_md = format_report(rep, f"({Path(args.input).name}, no_llm={bool(args.no_llm)})")
        (out_dir / "evaluation.md").write_text(report_md)
        write_error_analysis(rep.errors, pages, str(out_dir / "error_analysis.md"), methods)
        print(report_md)


def main() -> None:
    ap = argparse.ArgumentParser(prog="mk-doc-classifier")
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("classify", help="PDF 분류 파이프라인 실행")
    c.add_argument("--input", required=True)
    c.add_argument("--output", required=True)
    c.add_argument("--no-llm", action="store_true", help="룰 기반만으로 실행 (API 키 불필요)")
    c.add_argument("--gt", help="ground truth CSV 경로 (있으면 평가 수행)")
    c.set_defaults(func=run_classify)

    g = sub.add_parser("build-gt", help="testing_answers/ 원본 PDF에서 GT CSV 생성")
    g.set_defaults(func=lambda a: __import__("src.gt_builder", fromlist=["x"]).build_ground_truth())

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
