from src.extractor import PageRecord
from src.rule_classifier import classify_page


def make_page(text: str, head: str | None = None, page_no: int = 1) -> PageRecord:
    return PageRecord(
        page_no=page_no,
        text=text,
        head_text=head if head is not None else text[:200],
        rotation=0,
        n_chars=len(text),
        n_images=0,
    )


def test_urla_footer_signature_matches_in_body():
    # URLA 시그니처는 푸터(하단)에 있어 head에 없어도 본문 매칭으로 확정돼야 한다
    page = make_page(
        "Section 5: Declarations ...\n" * 20
        + "Uniform Residential Loan Application\nFreddie Mac Form 65 · Fannie Mae Form 1003",
        head="Section 5: Declarations ...",
    )
    r = classify_page(page)
    assert r.label == "URLA_1003"
    assert not r.needs_llm


def test_realtor_trap_does_not_trigger_income():
    # URLA 본문의 고용주명 "Veronica Salazar Realtor"가 INCOME_DOC으로 오분류되면 안 된다
    page = make_page(
        "Employer or Business Name: Veronica Salazar Realtor\n"
        "Uniform Residential Loan Application\nFreddie Mac Form 65"
    )
    r = classify_page(page)
    assert r.label == "URLA_1003"


def test_credit_report_xactus_header():
    page = make_page("370 Reed Rd., Suite 100 Broomall, PA 19008\nRepositories: TUC/EXP/EQX\nFICO")
    r = classify_page(page)
    assert r.label == "CREDIT_REPORT"
    assert not r.needs_llm


def test_title_exhibit_a_combo():
    page = make_page("EXHIBIT A\nLegal Description\nFor APN/Parcel ID(s): ...")
    r = classify_page(page)
    assert r.label == "TITLE_REPORT"


def test_low_signal_page_delegates_to_llm():
    page = make_page("[ Plat map removed in anonymized sample ]")
    r = classify_page(page)
    assert r.needs_llm


def test_irs_transcript_is_income():
    page = make_page("This Product Contains Sensitive Taxpayer Data\nWage and Income Transcript")
    r = classify_page(page)
    assert r.label == "INCOME_DOC"
    assert not r.needs_llm
