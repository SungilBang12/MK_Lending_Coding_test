from src.extractor import PageRecord
from src.grouper import group_documents, smooth_labels


def make_page(page_no, internal=None, total=None):
    return PageRecord(
        page_no=page_no, text="x", head_text="x", rotation=0, n_chars=1, n_images=0,
        internal_page=internal, internal_total=total,
    )


def test_label_and_total_disambiguate_colliding_page_numbers():
    # URLA와 CREDIT이 둘 다 "N of 2"여도 (label, M) 결합으로 분리돼야 한다
    pages = [
        make_page(1, 1, 2), make_page(2, 1, 2),
        make_page(3, 2, 2), make_page(4, 2, 2),
    ]
    labels = ["URLA_1003", "CREDIT_REPORT", "URLA_1003", "CREDIT_REPORT"]
    docs = group_documents(pages, labels)["logical_documents"]
    urla = next(d for d in docs if d["label"] == "URLA_1003")
    credit = next(d for d in docs if d["label"] == "CREDIT_REPORT")
    assert urla["reconstructed_order"] == [1, 3]
    assert credit["reconstructed_order"] == [2, 4]


def test_duplicate_internal_number_splits_into_two_docs():
    # 같은 문서 2부(1/2,2/2 × 2)가 섞여 있으면 인스턴스 2개로 분리
    pages = [make_page(1, 1, 2), make_page(2, 1, 2), make_page(3, 2, 2), make_page(4, 2, 2)]
    labels = ["TITLE_REPORT"] * 4
    docs = group_documents(pages, labels)["logical_documents"]
    assert len(docs) == 2
    assert all(d["n_pages"] == 2 for d in docs)


def test_unnumbered_pages_appended_to_same_label_doc():
    pages = [make_page(1, 1, 2), make_page(2), make_page(3, 2, 2)]
    labels = ["CREDIT_REPORT"] * 3
    docs = group_documents(pages, labels)["logical_documents"]
    assert len(docs) == 1
    assert docs[0]["reconstructed_order"] == [1, 3, 2]
    assert docs[0]["unnumbered_pages"] == [2]


def test_physical_segments_merge_adjacent():
    pages = [make_page(i) for i in range(1, 5)]
    labels = ["URLA_1003", "URLA_1003", "TITLE_REPORT", "URLA_1003"]
    segs = group_documents(pages, labels)["physical_segments"]
    assert segs == [
        {"label": "URLA_1003", "start_page": 1, "end_page": 2},
        {"label": "TITLE_REPORT", "start_page": 3, "end_page": 3},
        {"label": "URLA_1003", "start_page": 4, "end_page": 4},
    ]


def test_smoothing_only_low_conf_with_matching_total():
    pages = [make_page(1, 3, 9), make_page(2, 4, 9), make_page(3, 5, 9)]
    labels = ["CREDIT_REPORT", "OTHER", "CREDIT_REPORT"]
    # conf 높으면 보정 안 함
    out, log = smooth_labels(pages, labels, [0.9, 0.9, 0.9])
    assert out == labels and log == []
    # conf 낮고 internal_total 일치 → 보정 + 로그
    out, log = smooth_labels(pages, labels, [0.9, 0.3, 0.9])
    assert out == ["CREDIT_REPORT"] * 3
    assert log[0]["page_no"] == 2 and log[0]["to"] == "CREDIT_REPORT"


def test_smoothing_skips_when_total_mismatch():
    # 진짜 단독 페이지(다른 문서 체계)는 보정하지 않는다
    pages = [make_page(1, 3, 9), make_page(2, 1, 5), make_page(3, 4, 9)]
    labels = ["CREDIT_REPORT", "TITLE_REPORT", "CREDIT_REPORT"]
    out, log = smooth_labels(pages, labels, [0.9, 0.3, 0.9])
    assert out == labels and log == []
