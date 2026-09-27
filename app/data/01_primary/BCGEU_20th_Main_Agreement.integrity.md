# Vexilon Forensic Integrity Audit: BCGEU 20th Main Agreement

- **Source PDF SHA256:** `c25e13176e230224705613ea4e7a7ac3f85200fd38ac00bd2720e04667a0444b`
- **Extraction:** `pdftotext -layout` → `.tmp/main-20th-agreement.txt`
- **Conversion:** Automated structural markdown (`.tmp/convert_20th_to_md.py`); no LLM transcription.

## Comparison performed

1. **Token inventory (case-insensitive, length ≥ 4):** every distinct token in the `pdftotext` extract vs the markdown file. Gaps are listed below in sort order; they are *not* treated as missing contract language without manual review.
2. **Line-signature spot check:** normalized extract lines (≥ 24 characters, excluding TOC dot-leaders and page chrome) vs markdown lines. Reflow and heading markup cause most line signatures to differ; this check is diagnostic only.

| Metric | Extract | Markdown |
|--------|---------|----------|
| Distinct tokens (≥ 4 letters) | 3880 | 3876 |
| Tokens in extract absent from markdown | 4 | — |

## Ordered token discrepancies (extract → markdown)

Alphabetical list of distinct ≥4-letter tokens present in the extract but not in the markdown:

1. `asterisk`
2. `bold`
3. `burnaby`
4. `cariboo`
5. `castlegar`
6. `cranbrook`
7. `east`
8. `george`
9. `kamloops`
10. `kelowna`
11. `kootenay`
12. `langley`
13. `mainland`
14. `nanaimo`
15. `northwest`
16. `okanagan`
17. `peace`
18. `text`
19. `west`
20. `williams`

Notes:

- `asterisk`, `bold`, and `text` come from the Nineteenth→Twentieth change legend in the PDF front matter, not operative clauses.
- City and region tokens (`burnaby`, `kamloops`, …) reflect office-address labels split across layout columns in `pdftotext`; the markdown office block uses structured headings instead of repeating those tokens on separate lines.
