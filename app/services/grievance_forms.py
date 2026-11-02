"""Fill official BCGEU grievance AcroForm PDFs from Grievance Builder wizard facts."""

from __future__ import annotations

import re
from io import BytesIO
from pathlib import Path

import pymupdf

FORMS_DIR = Path(__file__).resolve().parent.parent / "public" / "docs" / "forms"

TEMPLATE_A = FORMS_DIR / "Grievance_-_A_-_Grievor_Case.pdf"
TEMPLATE_B = FORMS_DIR / "Grievance_-_B_-_Notify_Designates.pdf"
TEMPLATE_C = FORMS_DIR / "Grievance_-_C_-_Steward_Case.pdf"
TEMPLATE_INSTRUCTIONS = FORMS_DIR / "Grievance_-_0_-_Instructions.pdf"

DRAFT_FORM_NOTICE = (
    "**Draft grievance forms (editable PDFs — steward review required, not ready to file)**\n\n"
    "Download the partially filled forms below. Open them in a PDF reader to complete "
    "member contact fields and signatures before filing. Blank fields were left empty intentionally."
)

EMPTY_FACT_VALUES = frozenset({"skip", "none", "n/a", "na", "not sure", "unknown"})

ARTICLE_ONE_PATTERN = re.compile(r"\barticle\s*1\b", re.IGNORECASE)

DRAFT_STATEMENT_HEADINGS = (
    re.compile(r"^###\s*📝\s*Draft Statement of Grievance\s*$", re.IGNORECASE),
    re.compile(r"^###\s*Draft Statement of Grievance\s*$", re.IGNORECASE),
)


def _clean_fact(facts: dict[str, str], key: str) -> str:
    value = (facts.get(key) or "").strip()
    if not value or value.lower() in EMPTY_FACT_VALUES:
        return ""
    return value


def _format_articles(articles: str) -> str:
    if not articles:
        return ""
    lower = articles.lower()
    if "other relevant" in lower:
        return articles
    if ARTICLE_ONE_PATTERN.search(articles):
        return f"{articles}; and all other relevant articles"
    return f"{articles}; Article 1 and all other relevant articles"


def extract_draft_statement(rag_text: str) -> str:
    """Pull draft statement section from RAG analysis when present."""
    if not rag_text:
        return ""
    lines = rag_text.splitlines()
    capture = False
    collected: list[str] = []
    for line in lines:
        stripped = line.strip()
        if any(pattern.match(stripped) for pattern in DRAFT_STATEMENT_HEADINGS):
            capture = True
            continue
        if capture:
            if stripped.startswith("###"):
                break
            collected.append(line)
    return "\n".join(collected).strip()


def map_wizard_facts_to_pdf_fields(
    facts: dict[str, str],
    rag_analysis: str = "",
) -> dict[str, dict[str, str]]:
    """Map wizard fact ids to AcroForm field names per form template."""
    issue = _clean_fact(facts, "issue_summary")
    incident = _clean_fact(facts, "incident_date")
    articles = _format_articles(_clean_fact(facts, "articles"))
    management = _clean_fact(facts, "management_response")
    witnesses_raw = _clean_fact(facts, "witnesses_evidence")
    remedy = _clean_fact(facts, "remedy")

    details = extract_draft_statement(rag_analysis) or issue

    form_a = {
        "Date of violation": incident,
        "Article(s) Allegedly Violated": articles,
        "Details of Grievance": details,
        "Remedy Sought": remedy,
    }

    form_b: dict[str, str] = {}
    if management:
        form_b["Verbal response at Step 1"] = management

    form_c = {
        "What are the facts of the grievance?": issue,
        "When did the grievance occur? - Date": incident,
        "Part 1 - Why is this considered a grievance?": issue,
        "Part 2 - Employer response to grievance:\r": management,
        "Any Related Information\r": witnesses_raw,
    }

    return {"A": form_a, "B": form_b, "C": form_c}


def fill_acroform_pdf(template_path: Path, field_values: dict[str, str]) -> bytes:
    """Write field values into a PDF template without flattening the form."""
    doc = pymupdf.open(template_path)
    for page in doc:
        for widget in page.widgets() or []:
            name = widget.field_name
            if not name or name not in field_values:
                continue
            value = field_values[name]
            if not value:
                continue
            widget.field_value = value
            widget.update()
    buffer = BytesIO()
    doc.save(buffer, garbage=4, deflate=True)
    doc.close()
    return buffer.getvalue()


def form_b_has_mappable_data(form_b: dict[str, str]) -> bool:
    return any(v.strip() for v in form_b.values())


def count_pdf_form_fields(pdf_bytes: bytes) -> int:
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    count = 0
    for page in doc:
        count += len(list(page.widgets() or []))
    doc.close()
    return count


def build_filled_grievance_downloads(
    facts: dict[str, str],
    rag_analysis: str = "",
) -> list[tuple[str, str, bytes]]:
    """
    Return (filename, mime, content) tuples for Chainlit File elements.
    Includes instructions PDF (unchanged) and filled A/C (and B when data exists).
    """
    mapped = map_wizard_facts_to_pdf_fields(facts, rag_analysis=rag_analysis)
    downloads: list[tuple[str, str, bytes]] = []

    instructions = TEMPLATE_INSTRUCTIONS.read_bytes()
    downloads.append(
        ("Grievance_-_0_-_Instructions.pdf", "application/pdf", instructions),
    )

    downloads.append(
        (
            "Grievance_-_A_-_Grievor_Case_DRAFT.pdf",
            "application/pdf",
            fill_acroform_pdf(TEMPLATE_A, mapped["A"]),
        )
    )

    if form_b_has_mappable_data(mapped["B"]):
        downloads.append(
            (
                "Grievance_-_B_-_Notify_Designates_DRAFT.pdf",
                "application/pdf",
                fill_acroform_pdf(TEMPLATE_B, mapped["B"]),
            )
        )

    downloads.append(
        (
            "Grievance_-_C_-_Steward_Case_DRAFT.pdf",
            "application/pdf",
            fill_acroform_pdf(TEMPLATE_C, mapped["C"]),
        )
    )
    return downloads
