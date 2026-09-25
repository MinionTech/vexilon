"""
Unit and integration tests for sync_sources.py and sources.yaml.
Issue #695: Automate web-sourced document ingestion, provenance tracking, and drift detection.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch
import urllib.error

import pytest
from yaml.constructor import ConstructorError

from scripts.sync_sources import (
    HTMLContentExtractor,
    SelectorNotFoundError,
    SourceEntry,
    check_source_drift,
    clean_bclaws_content,
    compute_content_hash,
    extract_content,
    extract_substantive_body,
    format_provenance_header,
    load_registry,
    save_registry,
    sync_source,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
APP_ROOT = Path(__file__).resolve().parent.parent
SOURCES_YAML = APP_ROOT / "data" / "sources.yaml"


def test_sources_yaml_exists_and_valid():
    """Verify app/data/sources.yaml exists and parses cleanly into SourceEntry objects."""
    assert SOURCES_YAML.exists(), f"Missing {SOURCES_YAML}"
    entries = load_registry(SOURCES_YAML)
    assert len(entries) >= 3, f"Expected at least 3 source entries, got {len(entries)}"

    statute_selectors = {
        "app/data/02_statutory/BC_Labour_Relations_Code.md": "#contentsscroll",
        "app/data/02_statutory/BC_Employment_Standards_Act.md": "#contentsscroll",
        "app/data/02_statutory/BC_Human_Rights_Code.md": "#contentsscroll",
    }
    seen_statutes: set[str] = set()

    for entry in entries:
        assert entry.path.startswith("app/data/"), f"Path must be in app/data/: {entry.path}"
        assert entry.url.startswith("http"), f"Invalid URL: {entry.url}"
        assert entry.type in ("html_selector", "bclaws"), f"Unknown type: {entry.type}"
        assert entry.category in ("primary", "statutory", "resources", "jurisprudence")
        if entry.path in statute_selectors:
            assert entry.selector == statute_selectors[entry.path], entry.path
            seen_statutes.add(entry.path)
        # Ensure referenced file actually exists in repo
        target_file = REPO_ROOT / entry.path
        assert target_file.exists(), f"Target document does not exist: {target_file}"
        # The registry baseline is the committed file. A stale hash makes MATCH
        # impossible and the Sunday job files a drift issue while this test stays green.
        substantive = extract_substantive_body(target_file.read_text(encoding="utf-8"))
        assert entry.content_hash == compute_content_hash(substantive), (
            f"content_hash for {entry.path} does not match the committed file"
        )

    assert seen_statutes == set(statute_selectors)


def test_load_registry_rejects_duplicate_keys(tmp_path):
    """Regression: load_registry must raise ConstructorError on duplicate mapping keys.

    PyYAML's SafeLoader silently overwrites duplicate keys with the last value.
    A duplicate 'path' or 'content_hash' in sources.yaml would corrupt drift
    baselines without any visible error.  _StrictSafeLoader makes this a hard
    failure so operator typos are caught at load time.
    """
    bad_yaml = tmp_path / "sources.yaml"
    bad_yaml.write_text(
        "sources:\n"
        "  - path: app/data/test.md\n"
        "    path: app/data/other.md\n"  # duplicate key
        "    url: https://example.com/test\n"
        "    type: html_selector\n"
        "    category: primary\n",
        encoding="utf-8",
    )
    with pytest.raises(ConstructorError, match="duplicate key"):
        load_registry(bad_yaml)


def test_html_content_extractor_selector():
    """Verify HTMLContentExtractor isolates the target selector and strips layout chrome."""
    raw_html = """
    <!DOCTYPE html>
    <html>
    <head><title>Test Document Title</title></head>
    <body>
        <nav><a href="/home">Home</a></nav>
        <header><h1>Site Header Banner</h1></header>
        <div id="body">
            <h1>Substantive Document Heading</h1>
            <p>This is substantive policy paragraph with a <a href="https://example.com/ref">citation link</a>.</p>
            <ul>
                <li>Bullet item 1</li>
                <li>Bullet item 2 with <strong>bold</strong> and <em>italic</em></li>
            </ul>
        </div>
        <footer><p>Copyright 2026</p></footer>
    </body>
    </html>
    """
    extractor = HTMLContentExtractor(target_selector="#body")
    extractor.feed(raw_html)
    markdown = extractor.get_markdown()

    assert "Site Header Banner" not in markdown
    assert "Copyright 2026" not in markdown
    assert "# Substantive Document Heading" in markdown
    assert "[citation link](https://example.com/ref)" in markdown
    assert "- Bullet item 1" in markdown
    assert "**bold**" in markdown
    assert "*italic*" in markdown


def test_html_content_extractor_void_tags_do_not_corrupt_selector_depth():
    """Regression: void elements inside a selector must not corrupt selector_depth.

    Before the fix, <br> and self-closing void tags incremented selector_depth in
    handle_starttag but Python's HTMLParser never emits a matching handle_endtag,
    leaving selector_depth permanently positive and leaking footer/post-container
    content into the extracted body.  With XHTML <br/>, handle_startendtag calls
    both handle_starttag and handle_endtag; the paired decrement would decrement
    depth to 0 prematurely, closing the container early.
    """
    raw_html = """
    <html>
    <head><title>Void Tag Test</title></head>
    <body>
        <div id="body">
            <h1>Inside Section</h1>
            <p>First paragraph.<br>Second line after br.<br/>Third line after self-close.</p>
            <img src="logo.png" alt="logo">
            <hr>
            <p>Still inside the container.</p>
        </div>
        <p>This paragraph is OUTSIDE the container and must not appear.</p>
    </body>
    </html>
    """
    extractor = HTMLContentExtractor(target_selector="#body")
    extractor.feed(raw_html)
    markdown = extractor.get_markdown()

    assert "# Inside Section" in markdown
    assert "First paragraph." in markdown
    assert "Second line after br." in markdown
    assert "Third line after self-close." in markdown
    assert "Still inside the container." in markdown
    assert "This paragraph is OUTSIDE" not in markdown


def test_clean_bclaws_content():
    """Verify BCLaws specific cleaner removes script artifacts and watermarks."""
    raw_bclaws_html = """
    <div id="contentsscroll">
        <script>
        function launchNewWindow(url) { window.open(url); }
        window.onload = function() { document.body.style.display = "block"; }
        </script>
        <h1>Labour Relations Code</h1>
        <p>Section 1 (1) Definitions and interpretations.</p>
        <p>Source link: https://www.bclaws.gov.bc.ca/civix/document/id/complete/statreg/96244_01</p>
    </div>
    """
    cleaned = clean_bclaws_content(raw_bclaws_html, selector="#contentsscroll")
    assert "launchNewWindow" not in cleaned
    assert "window.onload" not in cleaned
    assert "https://www.bclaws.gov.bc.ca" not in cleaned
    assert "Labour Relations Code" in cleaned
    assert "Section 1 (1) Definitions" in cleaned


def test_extract_substantive_body_strips_provenance():
    """Verify extract_substantive_body strips provenance headers to make drift hash invariant to dates."""
    doc_with_header_v1 = (
        "# Document Title\n\n"
        "**Source:** [Doc](https://example.com)  \n"
        "**Upstream Last Modified:** 2024-08-19  \n"
        "**Ingestion Date:** 2026-09-18  \n\n"
        "## Substantive Section\n\n"
        "Important legal text here."
    )
    doc_with_header_v2 = (
        "# Document Title\n\n"
        "**Source:** [Doc](https://example.com)  \n"
        "**Upstream Last Modified:** 2024-08-19  \n"
        "**Ingestion Date:** 2026-10-01  \n\n"
        "## Substantive Section\n\n"
        "Important legal text here."
    )

    body_1 = extract_substantive_body(doc_with_header_v1)
    body_2 = extract_substantive_body(doc_with_header_v2)

    assert body_1 == body_2
    assert compute_content_hash(body_1) == compute_content_hash(body_2)
    assert "Ingestion Date" not in body_1
    assert "## Substantive Section\n\nImportant legal text here." in body_1


def test_format_provenance_header():
    """Verify provenance header structure satisfies Issue #695 format."""
    header = format_provenance_header(
        title="BC Standards of Conduct",
        url="https://example.com/standards",
        upstream_last_modified="Wed, 21 Aug 2024 12:00:00 GMT",
        ingestion_date="2026-09-18",
    )
    assert header.startswith("# BC Standards of Conduct\n\n")
    assert "**Source:** [BC Standards of Conduct](https://example.com/standards)  \n" in header
    assert "**Upstream Last Modified:** Wed, 21 Aug 2024 12:00:00 GMT  \n" in header
    assert "**Ingestion Date:** 2026-09-18  \n\n" in header


def test_format_provenance_header_collapses_wrapped_title():
    """A newline inside a BC Laws title must not split the provenance heading."""
    header = format_provenance_header(
        title="Table of Contents - Workers Compensation\n    Act",
        url="https://www.bclaws.gov.bc.ca/civix/document/id/complete/statreg/19001_00",
        upstream_last_modified="Fri, 25 Sep 2026 02:45:46 GMT",
        ingestion_date="2026-09-25",
    )
    assert header.startswith("# Table of Contents - Workers Compensation Act\n\n")
    assert (
        "**Source:** [Table of Contents - Workers Compensation Act]"
        "(https://www.bclaws.gov.bc.ca/civix/document/id/complete/statreg/19001_00)  \n"
    ) in header


@patch("scripts.sync_sources.fetch_upstream")
def test_check_source_drift_match(mock_fetch):
    """Verify check_source_drift reports MATCH when upstream matches local substantive hash."""
    entry = SourceEntry(
        path="app/data/03_resources/BC_Criminal_Notification_Procedures.md",
        url="https://example.com/procedures",
        type="html_selector",
        selector="#body",
    )
    local_path = REPO_ROOT / entry.path
    local_text = local_path.read_text(encoding="utf-8")
    local_substantive = extract_substantive_body(local_text)

    # Mock upstream HTML that renders the same substantive body
    mock_html = f"<html><body><div id='body'><p>{local_substantive}</p></div></body></html>"
    mock_fetch.return_value = (mock_html, {"last-modified": "Fri, 18 Sep 2026 00:00:00 GMT"})

    res = check_source_drift(entry, REPO_ROOT)
    assert res.status == "MATCH"
    assert res.error is None


@patch("scripts.sync_sources.fetch_upstream")
def test_check_source_drift_detected(mock_fetch):
    """Verify check_source_drift detects divergence when upstream changes."""
    entry = SourceEntry(
        path="app/data/03_resources/BC_Criminal_Notification_Procedures.md",
        url="https://example.com/procedures",
        type="html_selector",
        selector="#body",
    )
    mock_html = "<html><body><div id='body'><h1>Modified Policy</h1><p>Completely rewritten text.</p></div></body></html>"
    mock_fetch.return_value = (mock_html, {"last-modified": "Sat, 19 Sep 2026 10:00:00 GMT"})

    res = check_source_drift(entry, REPO_ROOT)
    assert res.status == "DRIFT_DETECTED"
    assert res.local_hash != res.upstream_hash


@patch("scripts.sync_sources.fetch_upstream")
def test_check_source_drift_network_error(mock_fetch):
    """Verify check_source_drift cleanly handles network failures without false drift alert."""
    entry = SourceEntry(
        path="app/data/03_resources/BC_Criminal_Notification_Procedures.md",
        url="https://example.com/procedures",
        type="html_selector",
        selector="#body",
    )
    mock_fetch.side_effect = urllib.error.URLError("Connection refused")

    res = check_source_drift(entry, REPO_ROOT)
    assert res.status == "ERROR"
    assert "Connection refused" in (res.error or "")


@patch("scripts.sync_sources.fetch_upstream")
def test_sync_source_dry_run(mock_fetch, tmp_path):
    """Verify dry-run returns success without writing to file."""
    mock_html = "<html><body><div id='body'><h1>Test Doc</h1><p>Substantive text.</p></div></body></html>"
    mock_fetch.return_value = (mock_html, {"last-modified": "2026-09-18"})

    entry = SourceEntry(
        path="test_doc.md",
        url="https://example.com/test",
        type="html_selector",
        selector="#body",
    )
    target_file = tmp_path / "test_doc.md"
    assert not target_file.exists()

    success, content_hash = sync_source(entry, tmp_path, dry_run=True)
    assert success is True
    assert not target_file.exists()


def test_format_drift_alert_markdown():
    """Verify markdown alert table is formatted properly for steward review."""
    from scripts.sync_sources import DriftResult, format_drift_alert_markdown

    results = [
        DriftResult(
            path="app/data/02_statutory/BC_Labour_Relations_Code.md",
            url="https://example.com/lrc",
            status="DRIFT_DETECTED",
            local_hash="abcdef123456",
            upstream_hash="789012abcdef",
            upstream_last_modified="Thu, 17 Sep 2026 14:00:00 GMT",
        ),
        DriftResult(
            path="app/data/01_primary/Gov_BC_Standards_of_Conduct.md",
            url="https://example.com/standards",
            status="MATCH",
            local_hash="111111222222",
            upstream_hash="111111222222",
        ),
    ]

    alert_md = format_drift_alert_markdown(results)
    assert "## Upstream Document Drift Detected" in alert_md
    assert "| `app/data/02_statutory/BC_Labour_Relations_Code.md` | [BC_Labour_Relations_Code.md](https://example.com/lrc) | `abcdef12` | `789012ab` | Thu, 17 Sep 2026 14:00:00 GMT |" in alert_md
    assert "Gov_BC_Standards_of_Conduct.md" not in alert_md  # Only drifted docs should be listed
    assert "Steward Action Required" in alert_md
    assert "python app/scripts/sync_sources.py --sync" in alert_md


def test_html_content_extractor_input_tag_does_not_ratchet_ignore_depth():
    """Regression C2: <input> in a page must not permanently ratchet ignore_depth.

    input was previously in IGNORABLE_TAGS which increments ignore_depth on
    handle_starttag but HTML5 void elements never fire handle_endtag, so
    ignore_depth stays >=1 forever after the first <input> and all subsequent
    content is silently dropped.
    """
    raw_html = """
    <html>
    <body>
        <form>
            <input type="search" placeholder="Search...">
        </form>
        <div id="body">
            <h1>Content Heading</h1>
            <p>This content must be visible after the input element.</p>
        </div>
    </body>
    </html>
    """
    extractor = HTMLContentExtractor(target_selector="#body")
    extractor.feed(raw_html)
    markdown = extractor.get_markdown()

    assert "# Content Heading" in markdown
    assert "This content must be visible after the input element." in markdown


def test_clean_bclaws_url_regex_preserves_markdown_link_parens():
    """Regression C3: bclaws URL regex must not eat the closing ) of a markdown link.

    The original \\S+ pattern would match ) and ] ending a [text](url) construct,
    destroying the link syntax and changing the substantive hash.
    """
    content_with_link = (
        "See [Labour Relations Code](https://www.bclaws.gov.bc.ca/civix/document/id/lc/statreg/96244_01) "
        "for full text."
    )
    result = clean_bclaws_content(
        f"<div id='contentsscroll'>{content_with_link}</div>",
        selector="#contentsscroll",
    )
    # The URL inside parens should be stripped, but the surrounding text preserved
    assert "See" in result
    assert "for full text." in result
    # The bclaws URL itself should be gone
    assert "bclaws.gov.bc.ca" not in result
    # The URL is stripped but the closing ) must survive, leaving an empty target.
    # Old buggy output (\\S+ regex): the ) was eaten, breaking the link entirely.
    # Fixed output: [Labour Relations Code]() for full text.
    assert "Labour Relations Code]() for full text." in result


def test_html_content_extractor_multi_node_anchor_text():
    """Regression S6: anchor text spanning multiple inline nodes must produce one valid link.

    Previously <a href="x">Hello <strong>World</strong></a> produced
    [Hello](x)**World** — the link closed on the first text node and the bold
    content was orphaned.  After buffering, the full anchor text is emitted as
    [Hello **World**](x) (or equivalent), with no orphaned tokens.
    """
    raw_html = """
    <html><body>
    <div id="body">
        <p>Click <a href="https://example.com">Hello <strong>World</strong></a> here.</p>
    </div>
    </body></html>
    """
    extractor = HTMLContentExtractor(target_selector="#body")
    extractor.feed(raw_html)
    markdown = extractor.get_markdown()

    # The link must emit the full joined anchor text, not just the first text node.
    # Old buggy output: [Hello](url)**World** — link closed on first data node,
    # "World" orphaned.  Fixed output: [Hello World](url).
    assert "[Hello World](https://example.com)" in markdown
    assert "Click" in markdown
    assert "here." in markdown


def test_html_content_extractor_emphasis_inside_anchor_is_balanced():
    """Regression: <a><em>..</em></a> emitted the emphasis markers outside the buffered link.

    bclaws cross-references are <a href><em>Act name</em></a>; the old output was
    ``**[Labour Relations Code](...)`` (two stray ``*`` before the link) and
    ``****[Licence](...)`` for <a><strong>..</strong></a>.
    """
    raw_html = """
    <html><body>
    <div id="body">
        <p>as defined in the <a href="/civix/lrc"><em>Labour Relations Code</em></a>;</p>
        <p><a href="/standards/Licence.html"><strong>Licence</strong></a></p>
        <p><strong><a href="/x">Bold link</a></strong> and <em>plain italic</em>.</p>
    </div>
    </body></html>
    """
    extractor = HTMLContentExtractor(target_selector="#body")
    extractor.feed(raw_html)
    markdown = extractor.get_markdown()

    assert "as defined in the [Labour Relations Code](/civix/lrc);" in markdown
    assert "[Licence](/standards/Licence.html)" in markdown
    assert "**[Bold link](/x)**" in markdown
    assert "*[" not in markdown.replace("**[Bold link](/x)**", "")
    assert "*plain italic*" in markdown
    for line in markdown.splitlines():
        assert line.count("*") % 2 == 0, line


def test_extract_content_selector_list_keeps_sibling_containers():
    """Regression: gov.bc.ca renders the page's Resources section in a sibling of #body.

    A comma-separated selector list must extract every listed container in document
    order, skip the chrome between them, and fail closed when any listed container
    is absent.
    """
    raw_html = """
    <html><body>
      <nav>Site menu</nav>
      <div id="body"><h2>Responsibilities</h2><p>Employees must comply.</p></div>
      <div class="topicPageNav">Previous / Next</div>
      <div id="cmf-ui-supplementary-content"><h2 class="banner">Resources</h2>
        <ul><li><a href="https://example.com/hr09.pdf">HR policy 09</a></li></ul>
      </div>
      <footer>Copyright</footer>
    </body></html>
    """
    _title, body = extract_content(raw_html, "html_selector", "#body, #cmf-ui-supplementary-content")
    assert body.index("Employees must comply.") < body.index("## Resources")
    assert "- [HR policy 09](https://example.com/hr09.pdf)" in body
    assert "Previous / Next" not in body
    assert "Site menu" not in body
    assert "Copyright" not in body

    nested_html = """
    <html><body>
      <div id="body"><p>Outer text.</p>
        <div id="resources"><h2>Resources</h2><p>Nested once.</p></div>
        <p>After nested.</p>
      </div>
      <footer>Copyright</footer>
    </body></html>
    """
    _title, nested_body = extract_content(nested_html, "html_selector", "#body, #resources")
    assert nested_body.count("Nested once.") == 1
    assert nested_body.index("Outer text.") < nested_body.index("Nested once.") < nested_body.index("After nested.")
    assert "Copyright" not in nested_body

    without_resources = raw_html.replace('id="cmf-ui-supplementary-content"', 'id="gone"')
    with pytest.raises(SelectorNotFoundError, match="#cmf-ui-supplementary-content"):
        extract_content(without_resources, "html_selector", "#body, #cmf-ui-supplementary-content")


def test_extract_substantive_body_returns_empty_string_on_empty_body():
    """Regression W2: extract_substantive_body must return '' when all lines are provenance header.

    Previously returned markdown_text.strip() (full text including header) on the
    empty fallback.  On --sync the input has no header; on --check the local file
    does.  When body is genuinely empty, both sides must return '' so hashes match.
    """
    provenance_only = (
        "# Some Document Title\n\n"
        "**Source:** [Title](https://example.com)  \n"
        "**Upstream Last Modified:** 2026-09-01  \n"
        "**Ingestion Date:** 2026-09-18  \n\n"
        "---\n"
    )
    result = extract_substantive_body(provenance_only)
    assert result == ""


def test_check_source_drift_returns_untracked_when_no_local_file_and_no_hash(tmp_path):
    """Regression S1: missing local file + no registry hash must return UNTRACKED, not MATCH."""
    entry = SourceEntry(
        path="app/data/nonexistent.md",
        url="https://example.com/nonexistent",
        type="html_selector",
        selector="#body",
        content_hash=None,
    )
    result = check_source_drift(entry, tmp_path)
    assert result.status == "UNTRACKED"


def test_format_drift_alert_markdown_includes_error_entries():
    """Regression W4: ERROR sources must appear in the alert body when drift and errors co-occur."""
    from scripts.sync_sources import DriftResult, format_drift_alert_markdown

    results = [
        DriftResult(
            path="app/data/02_statutory/BC_Labour_Relations_Code.md",
            url="https://example.com/lrc",
            status="DRIFT_DETECTED",
            local_hash="a" * 64,
            upstream_hash="b" * 64,
            upstream_last_modified="Mon, 22 Sep 2026 00:00:00 GMT",
        ),
        DriftResult(
            path="app/data/01_primary/Some_Policy.md",
            url="https://example.com/policy",
            status="ERROR",
            error="Network error: timed out",
        ),
    ]

    alert_md = format_drift_alert_markdown(results)
    assert "Sources That Could Not Be Reached" in alert_md
    assert "Some_Policy.md" in alert_md
    assert "timed out" in alert_md
    # Drifted source must still appear in the main table
    assert "BC_Labour_Relations_Code.md" in alert_md


def test_clean_bclaws_content_raises_when_selector_missing():
    """A missing container must not fall back to hashing the whole page."""
    raw_html = "<html><body><div id='toolBar'>Search Results</div><p>Copyright</p></body></html>"
    with pytest.raises(SelectorNotFoundError, match="#contentsscroll"):
        clean_bclaws_content(raw_html, selector="#contentsscroll")


def test_bclaws_body_uses_validated_selector_not_whole_page():
    """The body is the container the guard validated, not a second default or the whole page.

    A null registry selector must fail closed. An entry selector other than any
    historical default must be the container that is hashed.
    """
    raw_html = """
    <html><body>
      <div id="toolBar">Search Results Copyright banner</div>
      <div id="decoy"><p>Whole page decoy that must not be hashed.</p></div>
      <div id="act-text">
        <h1>Labour Relations Code</h1>
        <p>Section 1 Definitions.</p>
      </div>
    </body></html>
    """
    with pytest.raises(SelectorNotFoundError) as missing_clean:
        clean_bclaws_content(raw_html, selector=None)
    clean_error = str(missing_clean.value)
    assert clean_error.strip()
    assert "missing" in clean_error.lower()
    assert "selector" in clean_error.lower()

    with pytest.raises(SelectorNotFoundError) as missing_extract:
        extract_content(raw_html, "bclaws", None)
    extract_error = str(missing_extract.value)
    assert extract_error.strip()
    assert "missing" in extract_error.lower()
    assert "selector" in extract_error.lower()

    _title, body = extract_content(raw_html, "bclaws", "#act-text")
    assert "Section 1 Definitions." in body
    assert "Search Results" not in body
    assert "Whole page decoy" not in body


def test_save_registry_preserves_leading_comment_block(tmp_path):
    """--sync rewrites sources.yaml and must keep the schema comment header."""
    config = tmp_path / "sources.yaml"
    config.write_text(
        "# Declarative Source Registry\n"
        "# Schema comment that must survive --sync\n"
        "\n"
        "sources:\n"
        "  - path: app/data/test.md\n"
        "    url: https://example.com/test\n"
        "    type: html_selector\n"
        "    selector: '#body'\n"
        "    category: primary\n",
        encoding="utf-8",
    )
    entries = load_registry(config)
    entries[0].content_hash = "ab" * 32
    save_registry(config, entries)
    text = config.read_text(encoding="utf-8")
    assert text.startswith("# Declarative Source Registry\n")
    assert "# Schema comment that must survive --sync\n" in text
    reloaded = load_registry(config)
    assert reloaded[0].content_hash == "ab" * 32
    assert reloaded[0].selector == "#body"


@patch("scripts.sync_sources.fetch_upstream")
def test_missing_selector_is_error_and_does_not_overwrite(mock_fetch, tmp_path):
    """Selector miss is an operational error. --sync must not replace the committed file."""
    target = tmp_path / "app" / "data" / "statute.md"
    target.parent.mkdir(parents=True)
    original = "# Statute\n\n## Definitions\n\nThe act text.\n"
    target.write_text(original, encoding="utf-8")
    mock_fetch.return_value = (
        "<html><body><div id='toolBar'>Search Results</div><p>Copyright banner</p></body></html>",
        {},
    )
    entry = SourceEntry(
        path="app/data/statute.md",
        url="https://example.com/statute",
        type="bclaws",
        selector="#contentsscroll",
        content_hash="a" * 64,
    )

    result = check_source_drift(entry, tmp_path)
    assert result.status == "ERROR"
    assert "Selector not found" in (result.error or "")
    assert "Search Results" not in (result.error or "")

    success, message = sync_source(entry, tmp_path, dry_run=False)
    assert success is False
    assert "Selector not found" in message
    assert target.read_text(encoding="utf-8") == original


@patch("scripts.sync_sources.fetch_upstream")
def test_empty_upstream_body_is_error_not_drift(mock_fetch, tmp_path):
    """An extract with no substantive body is an operational error, not drift."""
    target = tmp_path / "app" / "data" / "doc.md"
    target.parent.mkdir(parents=True)
    target.write_text("# Title\n\nBody text that must stay.\n", encoding="utf-8")
    mock_fetch.return_value = (
        "<html><head><title>T</title></head><body><div id='body'><h1>T</h1></div></body></html>",
        {},
    )
    entry = SourceEntry(
        path="app/data/doc.md",
        url="https://example.com/doc",
        type="html_selector",
        selector="#body",
        content_hash="c" * 64,
    )
    result = check_source_drift(entry, tmp_path)
    assert result.status == "ERROR"
    assert "no substantive" in (result.error or "")
    assert "Body text that must stay." in target.read_text(encoding="utf-8")
