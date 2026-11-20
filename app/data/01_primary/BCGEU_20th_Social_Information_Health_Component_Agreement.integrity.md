# Vexilon Forensic Integrity Audit: BCGEU 20th Social, Information & Health Component Agreement

- **Source PDF:** https://agreements.bcgeu.ca/document/V0hwelFPdW1KQkk9 (42 pages, Docusign-signed, dated September 29, 2026), committed as `app/public/docs/BCGEU_20th_Social_Information_Health_Component_Agreement.pdf`
- **Source PDF SHA256:** `5d6854650ba808402af1a0b937f3eb2f8fa6d6f7d60cfbe345126f2469fe274b`
- **Extraction:** PyMuPDF `page.get_text()`; the signature page (PDF page 25) from `pdftotext -layout`, split into its two columns.
- **Conversion:** Automated structural markdown; no LLM transcription. Line breaks follow the PDF so lines can be matched to PDF pages.

## Structure

- `# ARTICLE n - TITLE` for Articles 1–14, `## n.n Title` for clauses, `# APPENDIX n - Title`, `# LETTER OF INTENT n - Title`, `# LETTER OF UNDERSTANDING n - Title` and `# MEMORANDUM OF UNDERSTANDING 1 - Title`. The ` - ` joining a number to its title is the form the PDF's table of contents uses.
- The Clause 3.12 (a)(1) work schedule table is a Markdown pipe table transcribed from the PDF layout. Its first column has no heading in the PDF and has none here.

## Removed

- Table of contents (PDF pages 3–5).
- Page furniture: the `Docusign Envelope ID` stamp, the `BCGEU and Social, Information & Health Component (03/2029)` running header, `Page n` numbers, roman page numbers, and the `260811v1`, `moveup`, `FA-639`, `Rev: 2021-09-24` and `Comp6-20th` document codes.
- Signature lines (underscores).

## Comparison performed

Normalized words (lowercase, punctuation removed) of the PDF body (all pages except the table of contents, page furniture removed) against the Markdown:

| Metric | PDF body | Markdown |
|--------|----------|----------|
| Words | 12859 | 12859 |
| Words in PDF absent from Markdown | 0 | — |
| Words in Markdown absent from PDF | — | 0 |
| Word-order similarity | 0.9960 | |

Word order differs only where the PDF's reading order is not the document's: the Clause 3.12 table (cells read row by row) and the signature page (union column, then employer column).

Result: PASS (word multiset identical).
