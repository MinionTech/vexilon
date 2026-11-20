# Vexilon Forensic Integrity Audit: BCGEU 20th Main Agreement Wage Scales (Appendix 3 to 3G)

- **Source PDF:** https://agreements.bcgeu.ca/document/cEFYT1pBS1g2NjQ9, *Main 20th BC Public Service Agency (Government of B.C.) - DRAFT - E&OE Copy* (235 pages, created August 31, 2026)
- **Source PDF SHA256:** `c25e13176e230224705613ea4e7a7ac3f85200fd38ac00bd2720e04667a0444b`. This is the same file the 20th Main Agreement Markdown was transcribed from.
- **Hosted excerpt:** PDF pages 121–160 (printed pages 109–148), committed as `app/public/docs/BCGEU_20th_Main_Agreement_Wage_Scales.pdf`. Citation page numbers are pages of this excerpt.
- **Extraction:** PyMuPDF words grouped into rows by position. Appendices 3D and 3F are printed sideways; they were transcribed from `pdftotext -layout` and checked against the page images.
- **Conversion:** Deterministic structural Markdown; no LLM transcription.

## Structure

- One table per grid level and effective date, under a `### Grid Level N (Effective date)` heading (`### Retail Stores & Warehouse Grid Level N (Effective date)` in Appendix 3G). The PDF prints the grid level once per block, beside step 3; the heading carries it for every row. Column headings are as printed on each page (`Bi-weekly`, or `Biweekly` on some Appendix 3G pages).
- Appendix 3C, 3D, 3F and the Appendix 3G classification list are tables with the PDF's column headings. The PDF repeats the table heading on every page; it appears once here.
- Appendix 3F: the PDF prints one grid and step entry for the cell holding Public Service Employment Program Level 2, Co-op Education Training Program Level 1 and the First Nations Forest Technicians Training Program. They share one row here. Program names are bold, as in the PDF.

## Removed

- Running header `BCGEU and Gov't of BC & BCPSA (03/2029)` and `Page n`.
- The end of Appendix 2 at the top of PDF page 121 and the start of Appendix 4 at the bottom of PDF page 160 (both in the 20th Main Agreement source).

## Comparison performed

Normalized words of PDF pages 121–160 (page furniture removed, Appendix 2 and 4 text excluded) against the Markdown:

| Metric | PDF | Markdown |
|--------|-----|----------|
| Money values (annual, monthly, bi-weekly, hourly) | 3271 | 3271 |
| Money values in the same order | yes | |
| PDF words absent from the Markdown | 55 | — |

The 55 absent words are the repeated per-page table headings (`CLASSIFICATION TITLE UNION COMP GRID RANGE` on 8 Appendix 3C pages; `Definition`, `Steps Used` and step numbers on the Appendix 3F pages) and three words the PDF hyphenates across lines in Appendix 3F (`post-` / `secondary`). Words present only in the Markdown are the grid headings and per-grid table headings described above.

## Values printed in error in the source

Values are transcribed as printed. A comparison of the April 6, 2025 and April 5, 2026 grids with the BC Public Service salary grids page (https://www2.gov.bc.ca/gov/content/careers-myhr/all-employees/pay-benefits/salaries/salarylookuptool/grids) matched 381 of 384 rows exactly. The three that differ are misprints in this PDF:

| Grid | Effective | Step | Column | PDF prints | gov.bc.ca |
|------|-----------|------|--------|------------|-----------|
| 2 | April 6, 2025 | 1 | Monthly | 3,593,91 | 3,593.91 |
| 10 | April 5, 2026 | 1 | Annual | 54,864.22 | 54,684.22 |
| 18 | April 5, 2026 | 4 | Annual | 74.009.08 | 74,009.08 |

The PDF's Appendix 3 has no grid levels 34 and 35 in the April 6, 2025 scale; they start April 5, 2026.
