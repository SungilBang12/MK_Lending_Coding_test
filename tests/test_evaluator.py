import csv

from src.evaluator import evaluate, load_gt_csv, segments_of


def test_segments_of_merges_adjacent():
    labels = ["A", "A", "B", "A", "A", "A"]
    assert segments_of(labels) == [("A", 1, 2), ("B", 3, 3), ("A", 4, 6)]


def test_evaluate_perfect():
    labels = ["URLA_1003", "URLA_1003", "CREDIT_REPORT"]
    rep = evaluate(labels, labels)
    assert rep.accuracy == 1.0
    assert rep.boundary_f1 == 1.0
    assert rep.doc_exact_match == 1.0
    assert rep.errors == []


def test_evaluate_boundary_vs_page_level():
    # 페이지 1개 오류가 세그먼트 지표에 더 크게 반영되는지 (NER token vs entity 관계)
    gt = ["URLA_1003"] * 3 + ["CREDIT_REPORT"] * 3
    pred = ["URLA_1003"] * 2 + ["CREDIT_REPORT"] * 4
    rep = evaluate(pred, gt)
    assert rep.accuracy == 5 / 6
    assert rep.doc_exact_match == 0.0  # 어느 세그먼트도 (라벨,시작,끝) 완전 일치 아님


def test_gt_csv_roundtrip(tmp_path):
    p = tmp_path / "gt.csv"
    rows = [
        {"page_no": "2", "label": "CREDIT_REPORT"},
        {"page_no": "1", "label": "URLA_1003"},
    ]
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["page_no", "label"])
        w.writeheader()
        w.writerows(rows)
    # page_no 순서로 정렬되어 로드돼야 한다
    assert load_gt_csv(str(p)) == ["URLA_1003", "CREDIT_REPORT"]
