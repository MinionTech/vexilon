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

# Fields that must never be auto-filled (PII / signatures / office use).
NEVER_AUTO_FILL_SUBSTRINGS = (
    "name",
    "address",
    "phone",
    "email",
    "signature",
    "postal",
    "city",
    "local",
    "employer",
    "branch",
    "bcgeu office",
    "clear",
    "print",
    "seniority",
    "job class",
    "work address",
    "designate name",
    "steward name",
    "steward email",
    "steward cell",
)


def _clean_fact(facts: dict[str, str], key: str) -> str:
    value = (facts.get(key) or "").strip()
    if not value or value.lower() == "skip":
        return ""
    return value


def _format_articles(articles: str) -> str:
    if not articles:
        return ""
    lower = articles.lower()
    if "article 1" in lower and "other relevant" in lower:
        return articles
    return f"{articles}; Article 1 and all other relevant articles"


def extract_draft_statement(rag_text: str) -> str:
    """Pull draft statement section from RAG analysis when present."""
    if not rag_text:
        return ""
    match = re.search(
        r"###\s*📝\s*Draft Statement of Grievance\s*\n(.*?)(?=\n###|\Z)",
        rag_text,
        re.DOTALL | re.IGNORECASE,
    )
    if match:
        return match.group(1).strip()
    match = re.search(
        r"###\s*Draft Statement of Grievance\s*\n(.*?)(?=\n###|\Z)",
        rag_text,
        re.DOTALL | re.IGNORECASE,
    )
    if match:
        return match.group(1).strip()
    return ""


def _parse_witness_lines(text: str) -> list[str]:
    if not text:
        return []
    lines = [ln.strip() for ln in re.split(r"[\n;]+", text) if ln.strip()]
    return lines[:3]


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
    witness_lines = _parse_witness_lines(witnesses_raw)

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
    witness_field_map = (
        ("Witness 1 - what they witnessed", 0),
        ("Witness 2 - what they witnessed", 1),
        ("Witness 3 - what they witnessed", 2),
    )
    for field_name, idx in witness_field_map:
        if idx < len(witness_lines):
            form_c[field_name] = witness_lines[idx]

    return {"A": form_a, "B": form_b, "C": form_c}


def _field_allowed(name: str) -> bool:
    lower = (name or "").lower()
    return not any(token in lower for token in NEVER_AUTO_FILL_SUBSTRINGS)


def fill_acroform_pdf(template_path: Path, field_values: dict[str, str]) -> bytes:
    """Write field values into a PDF template without flattening the form."""
    doc = pymupdf.open(template_path)
    for page in doc:
        for widget in page.widgets() or []:
            name = widget.field_name
            if not name or name not in field_values:
                continue
            value = field_values[name]
            if not value or not _field_allowed(name):
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
