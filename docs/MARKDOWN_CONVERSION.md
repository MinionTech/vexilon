# RAG Optimization: Converting PDF to Markdown

This document explains the rationale, benefits, and process for migrating the Vexilon knowledge base from legacy PDFs to high-fidelity Markdown.

## 1. Why Convert? (The Forensic Advantage)

While Vexilon's RAG pipeline includes advanced PDF extraction logic (`app.py`), PDFs are a "lossy" format for semantic search. Converting to Markdown (`.md`) provides three critical advantages:

### A. Semantic Hierarchy
*   **PDF Problem**: Sections and Articles are just text that "looks" like a header. The RAG has to guess whether "10.1" is a section start or just a number in a sentence.
*   **Markdown Solution**: Uses `# Article`, `## Section`, and `### Clause`. LLMs and embedding models (like BGE-small) are specifically trained to identify these headers, creating much stronger "context breadcrumbs" in the vector index.

### B. Noise Removal & Token Efficiency
*   **Web-to-PDF Artifacts**: Files from `bclaws.gov.bc.ca` contain date-stamps, footers, and URL lines that waste **15-30 tokens per chunk**.
*   **Clean MD**: Eliminates all UI noise, ensuring that 100% of the chunked tokens are substantive legal language. This means you can fit **more context** into the LLM's prompt for the same token cost.

### C. Reliability
*   **Zero Indexing Errors**: Markdown is plain text. There is no risk of column-overlap, character-encoding issues (ligatures), or page-number injection that can hallucinate contract articles.

## 2. Quantitative Benefits

| Feature | PDF (Raw) | Markdown (Optimized) |
| :--- | :--- | :--- |
| **Token Waste** | 10–20% (Noise) | <1% (Structure) |
| **Retrieval Accuracy** | Moderate | High (Header-aware) |
| **Chunk Continuity** | Page-bound | Section-bound |
| **Parsing Time** | Slow (OCR/Heuristics) | Instant |

## 3. How to Convert

We have provided a specialized script `scripts/pdf_to_md.py` that uses a **Forensic Dual-Pass** strategy. It uses **Claude 3.5 Sonnet** and **Haiku** to reconstruct the document with a "Zero-Reasoning" constraint.

### Safety Features
- **Dual-Pass Verification**: The script runs each page through two different models. If they disagree on the Markdown structure, it flags a divergence for your review.
- **Word-Fingerprinting**: The script extracts a "fingerprint" of substantive words from the raw PDF and compares it to the Markdown. It will alert you if the AI accidentally "hallucinates" a word that wasn't in the original text.
- **Strict Verbatim Prompt**: The AI is explicitly forbidden from rephrasing, "improving," or summarizing the source.

### Usage

1.  **Set Environment Variable**:
    ```bash
    export ANTHROPIC_API_KEY=<YOUR_ANTHROPIC_API_KEY>
    ```

2.  **Run the Converter**:
    ```bash
    # High-Integrity mode is ON by default
    python scripts/pdf_to_md.py "data/labour_law/01_primary/BC Labour Relations Code.pdf"
    ```

3.  **Audit the Result**:
    Quickly scan the generated `.md` file to ensure the header levels match the document's structure (e.g., `# ARTICLE 10`).

4.  **Re-index Vexilon**:
    After adding the `.md` file, force a rebuild of the vector index:
    ```bash
    # From project root
    python app.py --rebuild-index
    ```

## 4. Operational Best Practices
*   **Keep both for now**: Keep the PDF for "Download Original" links in the Gradio UI, but rely on the `.md` file for the AI's internal retrieval.
*   **Manual Touch-ups**: If a table is particularly complex, a quick manual edit of the `.md` file will permanently fix it for the RAG, something that is impossible with raw PDF extraction.
