"""Tests for BCGEU grievance AcroForm PDF filling."""

from pathlib import Path

import pymupdf

from services.grievance_forms import (
    ARTICLE_ONE_PATTERN,
    TEMPLATE_A,
    TEMPLATE_B,
    TEMPLATE_C,
    _format_articles,
    build_filled_grievance_downloads,
    count_pdf_form_fields,
    extract_draft_statement,
    fill_acroform_pdf,
    map_wizard_facts_to_pdf_fields,
)

SAMPLE_FACTS = {
    "issue_summary": "Supervisor denied scheduled leave without cause.",
    "incident_date": "2026-03-15",
    "articles": "Article 27",
    "management_response": "Claimed operational need at Step 1.",
    "witnesses_evidence": "Coworker on shift saw the denial; timesheet for that week",
    "remedy": "Make the member whole for the denied leave.",
}


def test_format_articles_does_not_match_article_10_as_article_1():
    result = _format_articles("Article 12 discipline")
    assert "Article 12" in result
    assert ARTICLE_ONE_PATTERN.search("Article 12 discipline") is None
    assert "Article 1 and all other" in result


def test_format_articles_skips_none():
    facts = dict(SAMPLE_FACTS)
    facts["articles"] = "none"
    mapped = map_wizard_facts_to_pdf_fields(facts)
    assert mapped["A"]["Article(s) Allegedly Violated"] == ""


def test_extract_draft_statement_empty_section_falls_back():
    rag = "### 📝 Draft Statement of Grievance\n\n### 🛡️ Rebuttal\nMgmt wrong."
    assert extract_draft_statement(rag) == ""


def test_form_c_includes_management_response_in_pdf():
    mapped = map_wizard_facts_to_pdf_fields(SAMPLE_FACTS)
    filled = fill_acroform_pdf(TEMPLATE_C, mapped["C"])
    doc = pymupdf.open(stream=filled, filetype="pdf")
    values = {}
    for page in doc:
        for widget in page.widgets() or []:
            if widget.field_name:
                values[widget.field_name] = widget.field_value
    doc.close()
    assert values.get("Part 2 - Employer response to grievance:\r") == SAMPLE_FACTS["management_response"]


def test_map_wizard_facts_to_form_a_fields():
    mapped = map_wizard_facts_to_pdf_fields(SAMPLE_FACTS, rag_analysis="")
    assert mapped["A"]["Date of violation"] == "2026-03-15"
    assert "Article 27" in mapped["A"]["Article(s) Allegedly Violated"]
    assert "Article 1" in mapped["A"]["Article(s) Allegedly Violated"]
    assert mapped["A"]["Remedy Sought"] == SAMPLE_FACTS["remedy"]
    assert "Name" not in mapped["A"]


def test_map_wizard_facts_includes_form_b_when_management_present():
    mapped = map_wizard_facts_to_pdf_fields(SAMPLE_FACTS)
    assert mapped["B"]["Verbal response at Step 1"] == SAMPLE_FACTS["management_response"]


def test_map_wizard_facts_omits_form_b_when_management_skipped():
    facts = dict(SAMPLE_FACTS)
    facts["management_response"] = "skip"
    mapped = map_wizard_facts_to_pdf_fields(facts)
    assert mapped["B"] == {}


def test_fill_acroform_pdf_preserves_editable_fields():
    mapped = map_wizard_facts_to_pdf_fields(SAMPLE_FACTS)
    template_count = count_pdf_form_fields(TEMPLATE_A.read_bytes())
    filled = fill_acroform_pdf(TEMPLATE_A, mapped["A"])
    assert count_pdf_form_fields(filled) == template_count
    doc = pymupdf.open(stream=filled, filetype="pdf")
    values = {}
    for page in doc:
        for widget in page.widgets() or []:
            if widget.field_name:
                values[widget.field_name] = widget.field_value
    doc.close()
    assert values.get("Remedy Sought") == SAMPLE_FACTS["remedy"]
    assert not values.get("Name")


def test_build_filled_grievance_downloads_includes_a_and_c_always():
    downloads = build_filled_grievance_downloads(SAMPLE_FACTS)
    names = [name for name, _, _ in downloads]
    assert "Grievance_-_A_-_Grievor_Case_DRAFT.pdf" in names
    assert "Grievance_-_C_-_Steward_Case_DRAFT.pdf" in names
    assert "Grievance_-_0_-_Instructions.pdf" in names


def test_build_filled_grievance_downloads_skips_b_without_data():
    facts = dict(SAMPLE_FACTS)
    facts["management_response"] = "skip"
    downloads = build_filled_grievance_downloads(facts)
    names = [name for name, _, _ in downloads]
    assert "Grievance_-_B_-_Notify_Designates_DRAFT.pdf" not in names


def test_build_filled_grievance_downloads_includes_b_with_data():
    downloads = build_filled_grievance_downloads(SAMPLE_FACTS)
    names = [name for name, _, _ in downloads]
    assert "Grievance_-_B_-_Notify_Designates_DRAFT.pdf" in names


def test_is_successful_grieve_draft_rejects_heading_only():
    from services.llm import is_successful_grieve_draft_response

    text = "### 📝 Draft Statement of Grievance\n\n### 🛡️ Rebuttal\nShort"
    assert is_successful_grieve_draft_response(text) is False


def test_templates_exist():
    for path in (TEMPLATE_A, TEMPLATE_B, TEMPLATE_C):
        assert Path(path).is_file()
