"""
Unit and integration tests for sync_sources.py and sources.yaml.
Issue #695: Automate web-sourced document ingestion, provenance tracking, and drift detection.
"""

from __future__ import annotations

import hashlib
import re
import urllib.error
import urllib.request
from email.message import EmailMessage
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from yaml.constructor import ConstructorError

from scripts import sync_sources
from scripts.sync_sources import (
    HTMLContentExtractor,
    NOT_MODIFIED_DETAIL,
    NotModified,
    PartNotFoundError,
    RegistryError,
    SelectorNotFoundError,
    SourceEntry,
    _local_substantive_hash,
    check_source_drift,
    clean_bclaws_content,
    compute_content_hash,
    extract_content,
    extract_substantive_body,
    extraction_fingerprint,
    fetch_upstream,
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
        assert entry.path.startswith("app/data/") or entry.path.startswith("app/public/docs/"), (
            f"Path must be in app/data/ or app/public/docs/: {entry.path}"
        )
        assert entry.url.startswith("http"), f"Invalid URL: {entry.url}"
        assert entry.type in ("html_selector", "bclaws", "pdf"), f"Unknown type: {entry.type}"
        assert entry.category in ("primary", "statutory", "resources", "jurisprudence")
        if entry.path in statute_selectors:
            assert entry.selector == statute_selectors[entry.path], entry.path
            seen_statutes.add(entry.path)
        # Ensure referenced file actually exists in repo
        target_file = REPO_ROOT / entry.path
        assert target_file.exists(), f"Target document does not exist: {target_file}"
        # The registry baseline is the committed file. A stale hash makes MATCH
        # impossible and the Sunday job files a drift issue while this test stays green.
        if entry.type == "pdf":
            assert entry.content_hash == hashlib.sha256(target_file.read_bytes()).hexdigest(), (
                f"content_hash for {entry.path} does not match the committed file"
            )
        else:
            substantive = extract_substantive_body(target_file.read_text(encoding="utf-8"))
            assert entry.content_hash == compute_content_hash(substantive), (
                f"content_hash for {entry.path} does not match the committed file"
            )

    assert seen_statutes == set(statute_selectors)


_PART_HEADING_LINE = re.compile(r"^(?:#{1,6}[ \t]+)?Part (\d+)(?:[ \t]+[—–-].*)?[ \t]*$")


def test_committed_part_files_open_on_their_own_part():
    """A part file's first part heading is that part, and it does not carry a sibling."""
    entries = load_registry(SOURCES_YAML)
    part_entries = [entry for entry in entries if re.search(r"_Part_\d+\.md$", entry.path)]
    assert len(part_entries) == 42
    url_counts: dict[str, int] = {}
    for entry in part_entries:
        url_counts[entry.url] = url_counts.get(entry.url, 0) + 1

    for entry in part_entries:
        match = re.search(r"_Part_(\d+)\.md$", entry.path)
        assert match is not None
        expected = str(int(match.group(1)))
        if url_counts[entry.url] > 1:
            assert entry.part == expected, entry.path
        else:
            assert entry.part is None, entry.path
        text = (REPO_ROOT / entry.path).read_text(encoding="utf-8")
        assert "Link to consolidated regulation (PDF)" not in text
        assert not re.search(r"\[Contents\]\([^)]+\)\s*\|", text), entry.path
        body = extract_substantive_body(text)
        headings = [
            heading.group(1)
            for line in body.splitlines()
            if (heading := _PART_HEADING_LINE.match(line))
        ]
        assert headings, entry.path
        assert headings[0] == expected, (entry.path, headings[:4])
        assert all(number == expected for number in headings), (entry.path, headings)


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


def _write_sources(tmp_path: Path, entries: list[dict[str, str]]) -> Path:
    lines = ["sources:"]
    for entry in entries:
        lines.append(f"  - path: {entry['path']}")
        lines.append(f"    url: {entry['url']}")
        lines.append(f"    type: {entry.get('type', 'bclaws')}")
        lines.append("    selector: '#contentsscroll'")
        if "part" in entry:
            lines.append(f"    part: '{entry['part']}'")
        lines.append("    category: statutory")
        if "content_hash" in entry:
            lines.append(f"    content_hash: {entry['content_hash']}")
    path = tmp_path / "sources.yaml"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_load_registry_requires_distinct_parts_for_a_shared_url(tmp_path):
    """Two files on one BC Laws document must name different parts."""
    shared = "https://www.bclaws.gov.bc.ca/civix/document/id/complete/statreg/296_97_01"
    ok = _write_sources(tmp_path, [
        {"path": "app/data/02_statutory/Part_01.md", "url": shared, "part": "1", "content_hash": "a" * 64},
        {"path": "app/data/02_statutory/Part_02.md", "url": shared, "part": "2", "content_hash": "b" * 64},
    ])
    loaded = load_registry(ok)
    assert [entry.part for entry in loaded] == ["1", "2"]

    same_part = _write_sources(tmp_path, [
        {"path": "app/data/02_statutory/Part_01.md", "url": shared, "part": "1", "content_hash": "a" * 64},
        {"path": "app/data/02_statutory/Part_02.md", "url": shared, "part": "1", "content_hash": "b" * 64},
    ])
    with pytest.raises(RegistryError, match="different part values"):
        load_registry(same_part)

    missing_part = _write_sources(tmp_path, [
        {"path": "app/data/02_statutory/Part_01.md", "url": shared, "content_hash": "a" * 64},
        {"path": "app/data/02_statutory/Part_02.md", "url": shared, "content_hash": "b" * 64},
    ])
    with pytest.raises(RegistryError, match="different part values"):
        load_registry(missing_part)


def test_load_registry_rejects_a_shared_content_hash(tmp_path):
    """Two entries must never share a drift baseline."""
    digest = "c" * 64
    path = _write_sources(tmp_path, [
        {
            "path": "app/data/02_statutory/Part_04.md",
            "url": "https://www.bclaws.gov.bc.ca/civix/document/id/complete/statreg/296_97_02",
            "content_hash": digest,
        },
        {
            "path": "app/data/02_statutory/Part_05.md",
            "url": "https://www.bclaws.gov.bc.ca/civix/document/id/complete/statreg/296_97_03",
            "content_hash": digest,
        },
    ])
    with pytest.raises(RegistryError, match="content_hash"):
        load_registry(path)


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


def test_clean_bclaws_content_strips_chrome_and_joins_block_text():
    """King's Printer chrome is dropped, and text is not glued or split across blocks."""
    raw_html = """
    <div id="contentsscroll">
      <div id="header"><table><tr>
        <td>Copyright © King's Printer,<br />Victoria, British Columbia, Canada</td>
        <td>
          <a href="/standards/Licence.html"><strong>Licence</strong></a>
          <strong></strong>
          <a href="/standards/Disclaimer.html"><strong>Disclaimer</strong></a>
        </td>
      </tr></table></div>
      <p><a href="/civix/document/id/complete/statreg/296_97_00_multi">View Complete Regulation</a></p>
      <table><tr>
        <td>B.C. Reg. 296/97<br /><span>Workers' Compensation Board</span></td>
        <td>Deposited September 8, 1997</td>
      </tr></table>
      <h5><strong><a href="/civix/document/id/complete/statreg/296_97_pit">Link to Point in Time</a></strong></h5>
      <h3>[RSBC 2019] CHAPTER
								1</h3>
      <p>The <strong>employer</strong> must pay.</p>
    </div>
    """
    cleaned = clean_bclaws_content(raw_html, selector="#contentsscroll")
    assert "King's Printer" not in cleaned
    assert "Licence" not in cleaned
    assert "Disclaimer" not in cleaned
    assert "View Complete" not in cleaned
    assert "Point in Time" not in cleaned
    assert "****" not in cleaned
    assert "BoardDeposited" not in cleaned
    assert "Workers' Compensation Board Deposited September 8, 1997" in cleaned
    assert "CHAPTER 1" in cleaned
    assert "CHAPTER\n" not in cleaned
    assert "**employer**" in cleaned
    assert "The **employer** must pay." in cleaned


def _extract_body(inner_html: str) -> str:
    extractor = HTMLContentExtractor(target_selector="#body")
    extractor.feed(f'<div id="body">{inner_html}</div>')
    return extractor.get_markdown()


def test_bordered_table_becomes_markdown_pipe_table():
    """Each row keeps its columns, so a reader can tell which value is which ear."""
    markdown = _extract_body("""
      <p>TABLE</p>
      <table border="1">
        <tr><td>Item</td><td>Range of Hearing Loss<br />
          (decibels)</td><td>Percentage for Ear Most Affected</td></tr>
        <tbody>
          <tr><td>1</td><td>0-34</td><td>0</td></tr>
          <tr><td>2</td><td>35-39</td><td>0.3</td></tr>
        </tbody>
      </table>
      <p>After the table.</p>
    """)
    assert markdown == (
        "TABLE\n\n"
        "| Item | Range of Hearing Loss<br>(decibels) | Percentage for Ear Most Affected |\n"
        "| --- | --- | --- |\n"
        "| 1 | 0-34 | 0 |\n"
        "| 2 | 35-39 | 0.3 |\n\n"
        "After the table."
    )


def test_markdown_table_escapes_pipe_inside_cell():
    markdown = _extract_body("""
      <table border="1">
        <tr><th>Signal</th><th>Meaning</th></tr>
        <tr><td>1 | 2 whistles</td><td>STOP</td></tr>
      </table>
    """)
    assert markdown.splitlines() == [
        "| Signal | Meaning |",
        "| --- | --- |",
        r"| 1 \| 2 whistles | STOP |",
    ]


def test_markdown_table_cell_closes_emphasis_before_a_line_break():
    """``**Label: **Text`` is not valid emphasis; the space must follow the marker."""
    markdown = _extract_body("""
      <table border="1">
        <tr><td>Signal</td><td>Meaning</td></tr>
        <tr><td colspan="2"><strong>b) Slow Signals:<br /></strong>Any regular signal.</td></tr>
      </table>
    """)
    assert markdown.splitlines()[-2:] == [
        "| **b) Slow Signals:** | |",
        "| Any regular signal. | |",
    ]


def test_markdown_table_cell_opens_emphasis_after_leading_space():
    """``** Description**`` is not valid emphasis; the space must precede the marker."""
    markdown = _extract_body("""
      <table border="1">
        <tr><td>Item</td><td>Column 1<br /><strong> Description of
          Disease</strong></td></tr>
        <tr><td>1</td><td>Poisoning by lead</td></tr>
      </table>
    """)
    assert markdown.splitlines()[0] == "| Item | Column 1<br>**Description of Disease** |"


def test_clean_bclaws_content_keeps_space_between_adjacent_bold_runs():
    """``** **`` between two bold runs is not empty bold; the words must stay apart."""
    raw_html = """
    <div id="contentsscroll">
      <table border="1">
        <tr><td><strong><strong>Column 2</strong> <strong>Minimum distance</strong></strong></td></tr>
        <tr><td>3 m</td></tr>
      </table>
    </div>
    """
    cleaned = clean_bclaws_content(raw_html, selector="#contentsscroll")
    assert "2Minimum" not in cleaned
    assert cleaned.splitlines()[0] == "| Column 2 Minimum distance |"


def test_clean_bclaws_content_keeps_bold_balanced_per_paragraph():
    raw_html = """
    <div id="contentsscroll">
      <p><strong>Table 3-1</strong></p><p><strong>Minimum Requirements</strong></p>
    </div>
    """
    cleaned = clean_bclaws_content(raw_html, selector="#contentsscroll")
    assert cleaned == "**Table 3-1**\n\n**Minimum Requirements**"


def test_br_packed_row_splits_into_paired_rows_table_26_2():
    """OHS Table 26-2 packs three signals into one row; each count must stay with its command."""
    raw_html = """
    <div id="contentsscroll">
      <table align="center" cellSpacing="0" cellPadding="3" border="1" class="tablestyle2">
        <caption><p>Table 26-2: Audible signals for vehicle operations</p></caption>
        <tbody><tr><td colname="c1" width="300">1 whistle<br /> 2 whistles<br /> 3 whistles</td>
        <td colname="c2" width="300">STOP<br /> BACK UP<br /> GO AHEAD</td></tr></tbody>
      </table>
    </div>
    """
    cleaned = clean_bclaws_content(raw_html, selector="#contentsscroll")
    assert cleaned.splitlines() == [
        "Table 26-2: Audible signals for vehicle operations",
        "",
        "| 1 whistle | STOP |",
        "| --- | --- |",
        "| 2 whistles | BACK UP |",
        "| 3 whistles | GO AHEAD |",
    ]


def test_br_packed_row_splits_part_20_soil_table_and_keeps_its_header():
    raw_html = """
    <div id="contentsscroll">
      <table align="center" cellSpacing="0" cellPadding="3" border="1" class="tablestyle2"><tbody>
        <tr><td colname="c1" align="center"><strong>Soil type</strong></td>
        <td colname="c2" align="center">Column 2<br /><strong>Description of soil</strong></td></tr>
        <tr><td colname="c1" align="center">A<br />B<br />C</td>
        <td colname="c2">hard and solid<br />likely to crack or crumble<br />soft, sandy, filled or loose</td></tr>
      </tbody></table>
    </div>
    """
    cleaned = clean_bclaws_content(raw_html, selector="#contentsscroll")
    assert cleaned.splitlines() == [
        "| **Soil type** | Column 2<br>**Description of soil** |",
        "| --- | --- |",
        "| A | hard and solid |",
        "| B | likely to crack or crumble |",
        "| C | soft, sandy, filled or loose |",
    ]


def test_br_segments_that_do_not_pair_stay_in_one_cell_joined_by_br():
    markdown = _extract_body("""
      <table border="1">
        <tr><td>Workers</td><td>Requirements</td></tr>
        <tr><td>2 — 9</td><td>• Basic first aid kit<br />• Basic first aid attendant</td></tr>
        <tr><td>10<br />or more</td><td>• Kit<br />• Room<br />• Attendant</td></tr>
      </table>
    """)
    assert markdown.splitlines()[2:] == [
        "| 2 — 9 | • Basic first aid kit<br>• Basic first aid attendant |",
        "| 10<br>or more | • Kit<br>• Room<br>• Attendant |",
    ]


def test_br_packed_row_closes_and_reopens_emphasis_per_row():
    markdown = _extract_body("""
      <table border="1">
        <tr><td>Signal</td><td>Meaning</td></tr>
        <tr><td><strong>1 SHORT<br />2 SHORT</strong></td><td>STOP<br />GO</td></tr>
      </table>
    """)
    assert markdown.splitlines()[2:] == [
        "| **1 SHORT** | STOP |",
        "| **2 SHORT** | GO |",
    ]


def test_br_packed_row_under_a_rowspan_is_not_split():
    markdown = _extract_body("""
      <table border="1">
        <tr><td>Group</td><td>Signal</td><td>Meaning</td></tr>
        <tr><td rowspan="2">Logging</td><td>1 SHORT</td><td>STOP</td></tr>
        <tr><td>2 SHORT<br />3 SHORT</td><td>GO<br />BACK</td></tr>
      </table>
    """)
    assert markdown.splitlines()[2:] == [
        "| Logging | 1 SHORT | STOP |",
        "| | 2 SHORT<br>3 SHORT | GO<br>BACK |",
    ]


def test_markdown_table_colspan_leaves_spanned_cells_empty():
    """A spanned value is written once; the columns after it stay aligned."""
    markdown = _extract_body("""
      <table border="1">
        <tr><td>Respirator type</td><td>Form</td><td>Protection factor</td></tr>
        <tr><td colspan="3">Air purifying</td></tr>
        <tr><td colspan="2">Half facepiece</td><td>10</td></tr>
      </table>
    """)
    assert markdown.splitlines() == [
        "| Respirator type | Form | Protection factor |",
        "| --- | --- | --- |",
        "| Air purifying | | |",
        "| Half facepiece | | 10 |",
    ]


def test_markdown_table_rowspan_leaves_covered_cells_empty():
    markdown = _extract_body("""
      <table border="1">
        <tr><td>Disease</td><td>Process</td></tr>
        <tr><td rowspan="2">Poisoning by lead</td><td>Smelting</td></tr>
        <tr><td>Soldering</td></tr>
      </table>
    """)
    assert markdown.splitlines() == [
        "| Disease | Process |",
        "| --- | --- |",
        "| Poisoning by lead | Smelting |",
        "| | Soldering |",
    ]


def test_table_nested_in_a_data_cell_is_flattened_into_that_cell():
    markdown = _extract_body("""
      <table border="1">
        <tr><td>Depth</td><td>Note</td></tr>
        <tr><td>1.2 m</td><td>See footnote
          <table border="0" class="fn"><tr><td>1</td><td>Minimum only.</td></tr></table>
        </td></tr>
      </table>
    """)
    assert markdown.splitlines() == [
        "| Depth | Note |",
        "| --- | --- |",
        "| 1.2 m | See footnote 1 Minimum only. |",
    ]


def test_borderless_layout_table_stays_flowing_text():
    markdown = _extract_body("""
      <table border="0"><tr><td>amount paid ÷ days worked</td></tr></table>
      <table><tr><td>where</td><td>amount paid</td><td>is the amount paid</td></tr></table>
    """)
    assert "|" not in markdown
    assert markdown == "amount paid ÷ days worked\n\nwhere amount paid is the amount paid"


_BUNDLE_HTML = """
<div id="contentsscroll">
  <h5><a href="/civix/pdf">Link to consolidated regulation (PDF)</a></h5>
  <p>Part 1 — Definitions</p>
  <p>Section 1.1 defines a workplace.</p>
  <p>Part 2 — Application</p>
  <p>Section 2.1 applies to every workplace.</p>
  <p>Part 3 — Rights and Responsibilities</p>
  <p>Section 3.1 a worker may refuse unsafe work.</p>
  <p>Part 33</p>
  <p>Sections 33.1 to 33.52 are repealed.</p>
  <p>Part 34 — Rope Access</p>
  <p>Section 34.1 defines an anchor.</p>
  <p>
    <a href="/civix/document/id/complete/statreg/296_97_00">Contents</a> |
    <a href="/civix/document/id/complete/statreg/296_97_01">Parts 1 to 3</a> |
    <a href="/civix/document/id/complete/statreg/296_97_02">Part 4</a>
  </p>
</div>
"""


def test_clean_bclaws_content_slices_a_bundle_page_by_part():
    """A shared BC Laws document keeps only the named part, and drops the nav rail."""
    whole = clean_bclaws_content(_BUNDLE_HTML, selector="#contentsscroll")
    assert "Section 1.1 defines a workplace." in whole
    assert "Section 34.1 defines an anchor." in whole
    assert "Link to consolidated regulation (PDF)" not in whole
    assert "[Contents](" not in whole
    assert " | " not in whole

    part_2 = clean_bclaws_content(_BUNDLE_HTML, selector="#contentsscroll", part="2")
    assert part_2.startswith("Part 2 — Application")
    assert "Section 2.1 applies to every workplace." in part_2
    assert "Section 1.1" not in part_2
    assert "Section 3.1" not in part_2
    assert "Part 1 —" not in part_2
    assert "Part 3 —" not in part_2

    part_1 = clean_bclaws_content(_BUNDLE_HTML, selector="#contentsscroll", part="1")
    assert part_1.startswith("Part 1 — Definitions")
    assert "Section 1.1 defines a workplace." in part_1
    assert "Section 2.1" not in part_1
    # Part 10 must not be selected by a prefix of Part 1.
    assert "Part 33" not in part_1

    part_33 = clean_bclaws_content(_BUNDLE_HTML, selector="#contentsscroll", part="33")
    assert part_33.startswith("Part 33")
    assert "Sections 33.1 to 33.52 are repealed." in part_33
    assert "Part 34 —" not in part_33
    assert "Section 34.1" not in part_33


def test_clean_bclaws_content_raises_when_part_heading_is_missing():
    """A missing part heading fails closed instead of keeping the rest of the bundle."""
    with pytest.raises(PartNotFoundError, match="Part 9"):
        clean_bclaws_content(_BUNDLE_HTML, selector="#contentsscroll", part="9")


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


def test_content_hash_ignores_bclaws_currency_date_line():
    """A date-only bump of the BC Laws currency line is not drift. Other text is."""
    statute = "Section 1 applies to a worker."
    amended = "Section 1 applies to a dependent."
    openings = (
        "This Act is current to",
        "This regulation is current to",
        "This consolidation is current to",
        "**This consolidation is current to",
    )
    for opening in openings:
        earlier = f"{opening} September 15, 2026\n\n{statute}\n"
        later = f"{opening} September 22, 2026\n\n{statute}\n"
        changed = f"{opening} September 22, 2026\n\n{amended}\n"
        assert compute_content_hash(earlier) == compute_content_hash(later)
        assert compute_content_hash(later) != compute_content_hash(changed)

    charge = (
        "minimize the potential for an electrical charge or current to "
        "unintentionally reach an explosive\n"
    )
    assert compute_content_hash(charge) != compute_content_hash(
        charge.replace("unintentionally", "intentionally")
    )


@patch("scripts.sync_sources.fetch_upstream")
def test_sync_keeps_bclaws_currency_line_and_hash_ignores_its_date(mock_fetch, tmp_path):
    """The currency line is written into the markdown. Only its date is outside the hash."""
    mock_fetch.return_value = (
        """
        <html><body><div id="contentsscroll">
          <h1>Example Act</h1>
          <p>This Act is current to September 22, 2026</p>
          <p>Section 1 applies.</p>
        </div></body></html>
        """,
        {"last-modified": "Sun, 27 Sep 2026 00:00:00 GMT"},
    )
    entry = SourceEntry(
        path="app/data/example_act.md",
        url="https://www.bclaws.gov.bc.ca/civix/document/id/complete/statreg/example",
        type="bclaws",
        selector="#contentsscroll",
        category="statutory",
    )
    success, digest = sync_source(entry, tmp_path, dry_run=False)
    assert success is True
    written = (tmp_path / entry.path).read_text(encoding="utf-8")
    assert "This Act is current to September 22, 2026" in written
    bumped = written.replace("September 22, 2026", "September 15, 2026")
    assert compute_content_hash(extract_substantive_body(written)) == compute_content_hash(
        extract_substantive_body(bumped)
    )
    assert digest == compute_content_hash(extract_substantive_body(written))
    amended = written.replace("Section 1 applies.", "Section 1 does not apply.")
    assert digest != compute_content_hash(extract_substantive_body(amended))


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


@patch("scripts.sync_sources.generate_manifest")
@patch("scripts.sync_sources.fetch_upstream")
def test_sync_with_filter_keeps_every_registry_entry(mock_fetch, mock_manifest, tmp_path, monkeypatch):
    """Regression: --sync --filter must update only matched entries and save the full registry."""
    names = ("alpha", "beta", "gamma")
    urls = [f"https://example.com/doc-{name}" for name in names]
    config = tmp_path / "sources.yaml"
    config.write_text(
        "# Registry header\n"
        "\n"
        "sources:\n"
        + "".join(
            f"  - path: app/data/{name}.md\n"
            f"    url: {url}\n"
            "    type: html_selector\n"
            "    selector: '#body'\n"
            "    category: primary\n"
            f"    content_hash: {digit * 64}\n"
            "    last_synced: '2000-01-01'\n"
            for name, url, digit in zip(names, urls, "abc")
        ),
        encoding="utf-8",
    )
    before = load_registry(config)
    mock_fetch.return_value = (
        "<html><body><div id='body'><h1>Beta</h1><p>Fresh beta text.</p></div></body></html>",
        {},
    )
    monkeypatch.setattr(sync_sources, "_REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        "sys.argv",
        ["sync_sources.py", "--config", str(config), "--sync", "--filter", "doc-beta"],
    )

    assert sync_sources.main() == 0

    mock_fetch.assert_called_once_with(urls[1])
    mock_manifest.assert_called_once()
    after = load_registry(config)
    assert [entry.path for entry in after] == [entry.path for entry in before]
    assert after[0] == before[0]
    assert after[2] == before[2]
    assert after[1].content_hash != before[1].content_hash
    assert after[1].last_synced != before[1].last_synced
    assert after[1].url == before[1].url
    assert after[1].selector == before[1].selector
    assert config.read_text(encoding="utf-8").startswith("# Registry header\n")


_KEPT_BODY = "# Title\n\nKept body.\n"
_STALE_VALIDATOR = "Wed, 21 Aug 2024 12:00:00 GMT"


def _digest(body: str) -> str:
    return compute_content_hash(extract_substantive_body(body))


def _write_doc(tmp_path: Path, body: str = _KEPT_BODY) -> tuple[Path, str]:
    target = tmp_path / "app" / "data" / "doc.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(body, encoding="utf-8")
    return target, _digest(body)


def _conditional_registry(
    tmp_path: Path,
    etag: str,
    last_modified: str,
    *,
    body: str = _KEPT_BODY,
    content_hash: str | None = None,
) -> Path:
    """Registry whose content_hash is the substantive hash of ``body`` on disk."""
    _target, digest = _write_doc(tmp_path, body)
    fingerprint = extraction_fingerprint(SourceEntry(
        path="app/data/doc.md",
        url="https://example.com/doc",
        type="html_selector",
        selector="#body",
    ))
    config = tmp_path / "sources.yaml"
    config.write_text(
        "# Registry header\n"
        "\n"
        "sources:\n"
        "  - path: app/data/doc.md\n"
        "    url: https://example.com/doc\n"
        "    type: html_selector\n"
        "    selector: '#body'\n"
        "    category: primary\n"
        f"    content_hash: {content_hash or digest}\n"
        "    last_synced: '2000-01-01'\n"
        f"    etag: '{etag}'\n"
        f"    upstream_last_modified: '{last_modified}'\n"
        f"    extraction_fingerprint: {fingerprint}\n",
        encoding="utf-8",
    )
    return config


def _http_304(url: str = "https://example.com/doc") -> urllib.error.HTTPError:
    hdrs = EmailMessage()
    hdrs["ETag"] = '"unsolicited"'
    body = Mock()
    body.read.side_effect = AssertionError("body was read")
    return urllib.error.HTTPError(url, 304, "Not Modified", hdrs, body)


class _UrlResponse:
    def __init__(self, payload: bytes, headers: dict[str, str]) -> None:
        self._payload = payload
        self.headers = headers

    def read(self) -> bytes:
        return self._payload

    def __enter__(self) -> "_UrlResponse":
        return self

    def __exit__(self, *_args: object) -> bool:
        return False


def _html_paragraph(text: str) -> str:
    return f"<html><body><div id='body'><p>{text}</p></div></body></html>"


def _request_headers(req: urllib.request.Request) -> dict[str, str]:
    return {key.lower(): value for key, value in req.header_items()}


def _assert_unconditional(req: urllib.request.Request) -> None:
    sent = _request_headers(req)
    assert "if-none-match" not in sent
    assert "if-modified-since" not in sent


def _install_urlopen(monkeypatch, responder) -> None:
    def urlopen(req: urllib.request.Request, timeout: int) -> object:
        assert timeout == 15
        return responder(req)

    monkeypatch.setattr(sync_sources.urllib.request, "urlopen", urlopen)


def test_fetch_upstream_sends_conditional_headers(monkeypatch):
    """--check and --sync send stored validators and still read a 200 body."""
    captured: dict[str, object] = {}

    class _Resp:
        headers = {
            "ETag": '"v2"',
            "Last-Modified": "Thu, 01 Oct 2026 00:00:00 GMT",
        }

        def read(self) -> bytes:
            return b"<html></html>"

        def __enter__(self) -> "_Resp":
            return self

        def __exit__(self, *_args: object) -> bool:
            return False

    def urlopen(req: urllib.request.Request, timeout: int) -> _Resp:
        captured["timeout"] = timeout
        captured["headers"] = {key.lower(): value for key, value in req.header_items()}
        return _Resp()

    monkeypatch.setattr(sync_sources.urllib.request, "urlopen", urlopen)
    content, headers = fetch_upstream(
        "https://example.com/doc",
        etag='"v1"',
        upstream_last_modified="Wed, 21 Aug 2024 12:00:00 GMT",
    )
    sent = captured["headers"]
    assert isinstance(sent, dict)
    assert sent["if-none-match"] == '"v1"'
    assert sent["if-modified-since"] == "Wed, 21 Aug 2024 12:00:00 GMT"
    assert captured["timeout"] == 15
    assert content == "<html></html>"
    assert headers["etag"] == '"v2"'
    assert headers["last-modified"] == "Thu, 01 Oct 2026 00:00:00 GMT"


def test_fetch_upstream_304_does_not_read_body(monkeypatch):
    """A 304 response is not parsed. Zero body bytes are read."""
    body = Mock()
    body.read.side_effect = AssertionError("body was read")
    hdrs = EmailMessage()
    hdrs["ETag"] = '"same"'
    error = urllib.error.HTTPError(
        "https://example.com/doc",
        304,
        "Not Modified",
        hdrs,
        body,
    )

    def urlopen(_req: urllib.request.Request, timeout: int) -> object:
        assert timeout == 15
        raise error

    monkeypatch.setattr(sync_sources.urllib.request, "urlopen", urlopen)
    with pytest.raises(NotModified) as raised:
        fetch_upstream("https://example.com/doc", etag='"same"')
    body.read.assert_not_called()
    assert raised.value.headers["etag"] == '"same"'


def test_fetch_upstream_304_without_validators_stays_an_error(monkeypatch):
    """An unsolicited 304 is not NotModified. The body is still not parsed."""
    error = _http_304()

    def urlopen(_req: urllib.request.Request, timeout: int) -> object:
        assert timeout == 15
        raise error

    monkeypatch.setattr(sync_sources.urllib.request, "urlopen", urlopen)
    with pytest.raises(urllib.error.HTTPError) as raised:
        fetch_upstream("https://example.com/doc")
    assert raised.value.code == 304
    assert not isinstance(raised.value, NotModified)
    error.fp.read.assert_not_called()


@patch("scripts.sync_sources.extract_content")
@patch("scripts.sync_sources.fetch_upstream")
def test_check_304_is_match_with_zero_bytes_parsed(mock_fetch, mock_extract, tmp_path):
    """HTTP 304 is a MATCH only when the local file is still the validated baseline."""
    target, digest = _write_doc(tmp_path)
    mock_fetch.side_effect = NotModified({"etag": '"same"'})
    entry = SourceEntry(
        path="app/data/doc.md",
        url="https://example.com/doc",
        type="html_selector",
        selector="#body",
        content_hash=digest,
        etag='"same"',
        upstream_last_modified=_STALE_VALIDATOR,
    )
    entry.extraction_fingerprint = extraction_fingerprint(entry)

    result = check_source_drift(entry, tmp_path)

    assert result.status == "MATCH"
    assert result.error is None
    assert result.local_hash == digest
    assert result.upstream_hash == digest
    mock_extract.assert_not_called()
    mock_fetch.assert_called_once_with(
        "https://example.com/doc",
        etag='"same"',
        upstream_last_modified=_STALE_VALIDATOR,
    )
    assert entry.etag == '"same"'
    assert target.read_text(encoding="utf-8") == _KEPT_BODY


@patch("scripts.sync_sources.fetch_upstream")
def test_check_200_still_compares_hash_when_conditional_request_is_ignored(mock_fetch, tmp_path):
    """A 200 after validators are sent still extracts the body and compares hashes."""
    body = "# Title\n\nOriginal body.\n"
    _target, digest = _write_doc(tmp_path, body)
    mock_fetch.return_value = (
        "<html><body><div id='body'><h1>Title</h1><p>Rewritten body.</p></div></body></html>",
        {"etag": '"v2"', "last-modified": "Thu, 01 Oct 2026 00:00:00 GMT"},
    )
    entry = SourceEntry(
        path="app/data/doc.md",
        url="https://example.com/doc",
        type="html_selector",
        selector="#body",
        content_hash=digest,
        etag='"v1"',
        upstream_last_modified=_STALE_VALIDATOR,
    )
    entry.extraction_fingerprint = extraction_fingerprint(entry)

    result = check_source_drift(entry, tmp_path)

    assert result.status == "DRIFT_DETECTED"
    mock_fetch.assert_called_once_with(
        "https://example.com/doc",
        etag='"v1"',
        upstream_last_modified=_STALE_VALIDATOR,
    )
    assert entry.etag == '"v1"'


def _entry_with_validators(digest: str) -> SourceEntry:
    return SourceEntry(
        path="app/data/doc.md",
        url="https://example.com/doc",
        type="html_selector",
        selector="#body",
        content_hash=digest,
        etag='"same"',
        upstream_last_modified=_STALE_VALIDATOR,
    )


def test_check_edited_file_is_drift_on_real_200(monkeypatch, tmp_path):
    """An edited file is fetched unconditionally. A matching upstream is still drift."""
    target, digest = _write_doc(tmp_path)
    edited = "# Title\n\nEdited body.\n"
    target.write_text(edited, encoding="utf-8")
    html = _html_paragraph("Kept body.")
    _title, body = extract_content(html, "html_selector", "#body")
    assert compute_content_hash(extract_substantive_body(body)) == digest

    def responder(req: urllib.request.Request) -> _UrlResponse:
        _assert_unconditional(req)
        return _UrlResponse(html.encode("utf-8"), {"ETag": '"v2"'})

    _install_urlopen(monkeypatch, responder)
    result = check_source_drift(_entry_with_validators(digest), tmp_path)

    assert result.status == "DRIFT_DETECTED"
    assert result.local_hash == _digest(edited)
    assert result.upstream_hash == digest
    assert result.local_hash != result.upstream_hash


def test_check_deleted_file_is_drift_when_upstream_matches_registry(monkeypatch, tmp_path):
    """A 200 whose hash matches the registry is still drift when the file is gone."""
    target, digest = _write_doc(tmp_path)
    target.unlink()
    html = _html_paragraph("Kept body.")
    _title, body = extract_content(html, "html_selector", "#body")
    assert compute_content_hash(extract_substantive_body(body)) == digest

    def responder(req: urllib.request.Request) -> _UrlResponse:
        _assert_unconditional(req)
        return _UrlResponse(html.encode("utf-8"), {"Last-Modified": _STALE_VALIDATOR})

    _install_urlopen(monkeypatch, responder)
    result = check_source_drift(_entry_with_validators(digest), tmp_path)

    assert result.status == "DRIFT_DETECTED"
    assert result.status != "MATCH"
    assert result.local_hash is None
    assert result.upstream_hash == digest
    assert result.error == "Local file is absent"


def test_check_deleted_file_still_reports_upstream_drift(monkeypatch, tmp_path):
    """A missing file still compares the registry hash, so a changed upstream is drift."""
    target, digest = _write_doc(tmp_path)
    target.unlink()
    html = _html_paragraph("Rewritten body.")

    def responder(req: urllib.request.Request) -> _UrlResponse:
        _assert_unconditional(req)
        return _UrlResponse(html.encode("utf-8"), {})

    _install_urlopen(monkeypatch, responder)
    result = check_source_drift(_entry_with_validators(digest), tmp_path)

    assert result.status == "DRIFT_DETECTED"
    assert result.local_hash is None
    assert result.upstream_hash != digest
    assert result.error == "Local file is absent"


def test_check_unsolicited_304_is_error_not_match(monkeypatch, tmp_path):
    """A 304 to a request that sent no validators stays a failure, not MATCH."""
    _target, digest = _write_doc(tmp_path)
    error = _http_304()

    def urlopen(req: urllib.request.Request, timeout: int) -> object:
        sent = {key.lower(): value for key, value in req.header_items()}
        assert "if-none-match" not in sent
        assert "if-modified-since" not in sent
        assert timeout == 15
        raise error

    monkeypatch.setattr(sync_sources.urllib.request, "urlopen", urlopen)
    entry = SourceEntry(
        path="app/data/doc.md",
        url="https://example.com/doc",
        type="html_selector",
        selector="#body",
        content_hash=digest,
    )

    result = check_source_drift(entry, tmp_path)

    assert result.status == "ERROR"
    assert result.status != "MATCH"
    error.fp.read.assert_not_called()


@patch("scripts.sync_sources.generate_manifest")
@patch("scripts.sync_sources.extract_content")
@patch("scripts.sync_sources.fetch_upstream")
def test_sync_304_skips_parse_and_leaves_registry_unchanged(
    mock_fetch, mock_extract, mock_manifest, tmp_path, monkeypatch
):
    """304 during --sync does not parse HTML and does not rewrite sources.yaml."""
    config = _conditional_registry(
        tmp_path, etag='"v1"', last_modified="Wed, 21 Aug 2024 12:00:00 GMT"
    )
    before = config.read_text(encoding="utf-8")
    mock_fetch.side_effect = NotModified({"etag": '"v1"'})
    monkeypatch.setattr(sync_sources, "_REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        "sys.argv",
        ["sync_sources.py", "--config", str(config), "--sync"],
    )

    assert sync_sources.main() == 0

    mock_extract.assert_not_called()
    mock_manifest.assert_not_called()
    assert config.read_text(encoding="utf-8") == before
    mock_fetch.assert_called_once_with(
        "https://example.com/doc",
        etag='"v1"',
        upstream_last_modified="Wed, 21 Aug 2024 12:00:00 GMT",
    )


@patch("scripts.sync_sources.generate_manifest")
@patch("scripts.sync_sources.fetch_upstream")
def test_sync_200_updates_etag_and_upstream_last_modified(
    mock_fetch, mock_manifest, tmp_path, monkeypatch
):
    """A 200 during --sync stores the new ETag and Last-Modified in sources.yaml."""
    config = _conditional_registry(
        tmp_path, etag='"v1"', last_modified="Wed, 21 Aug 2024 12:00:00 GMT"
    )
    mock_fetch.return_value = (
        "<html><body><div id='body'><h1>Doc</h1><p>Fresh text.</p></div></body></html>",
        {"etag": '"v2"', "last-modified": "Thu, 01 Oct 2026 00:00:00 GMT"},
    )
    monkeypatch.setattr(sync_sources, "_REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        "sys.argv",
        ["sync_sources.py", "--config", str(config), "--sync"],
    )

    assert sync_sources.main() == 0

    mock_fetch.assert_called_once_with(
        "https://example.com/doc",
        etag='"v1"',
        upstream_last_modified="Wed, 21 Aug 2024 12:00:00 GMT",
    )
    mock_manifest.assert_called_once()
    after = load_registry(config)
    assert [entry.type for entry in after] == ["html_selector"]
    assert after[0].etag == '"v2"'
    assert after[0].upstream_last_modified == "Thu, 01 Oct 2026 00:00:00 GMT"
    assert after[0].content_hash != _digest(_KEPT_BODY)
    assert after[0].extraction_fingerprint == extraction_fingerprint(after[0])
    text = config.read_text(encoding="utf-8")
    assert text.startswith("# Registry header\n")


@patch("scripts.sync_sources.extract_content")
@patch("scripts.sync_sources.fetch_upstream")
def test_sync_304_does_not_write_the_document(mock_fetch, mock_extract, tmp_path):
    """304 leaves an intact local markdown untouched and reports the not-modified detail."""
    target, digest = _write_doc(tmp_path)
    mock_fetch.side_effect = NotModified({})
    entry = SourceEntry(
        path="app/data/doc.md",
        url="https://example.com/doc",
        type="bclaws",
        selector="#contentsscroll",
        content_hash=digest,
        etag='"same"',
        upstream_last_modified=_STALE_VALIDATOR,
    )
    entry.extraction_fingerprint = extraction_fingerprint(entry)

    success, detail = sync_source(entry, tmp_path, dry_run=False)

    assert success is True
    assert detail == NOT_MODIFIED_DETAIL
    mock_extract.assert_not_called()
    assert target.read_text(encoding="utf-8") == _KEPT_BODY
    assert entry.etag == '"same"'
    assert entry.content_hash == digest
    assert entry.type == "bclaws"
    mock_fetch.assert_called_once_with(
        "https://example.com/doc",
        etag='"same"',
        upstream_last_modified=_STALE_VALIDATOR,
    )


def test_sync_edited_local_fetches_unconditionally_and_rewrites(monkeypatch, tmp_path):
    """An edited local file is fetched with no validators and rewritten from the 200."""
    target, digest = _write_doc(tmp_path)
    target.write_text("# Title\n\nEdited body.\n", encoding="utf-8")
    html = _html_paragraph("Restored from upstream.")
    entry = _entry_with_validators(digest)

    def responder(req: urllib.request.Request) -> _UrlResponse:
        _assert_unconditional(req)
        return _UrlResponse(
            html.encode("utf-8"),
            {"ETag": '"v2"', "Last-Modified": "Thu, 01 Oct 2026 00:00:00 GMT"},
        )

    _install_urlopen(monkeypatch, responder)
    success, _detail = sync_source(entry, tmp_path, dry_run=False)

    assert success is True
    written = target.read_text(encoding="utf-8")
    assert "Restored from upstream." in written
    assert "Edited body." not in written
    assert entry.content_hash != digest


def test_sync_deleted_local_304_does_not_succeed(monkeypatch, tmp_path):
    """A 304 on the unconditional fetch of a missing file is a failure, not a restore."""
    target, digest = _write_doc(tmp_path)
    target.unlink()
    error = _http_304()

    def responder(req: urllib.request.Request) -> object:
        _assert_unconditional(req)
        raise error

    _install_urlopen(monkeypatch, responder)
    success, detail = sync_source(_entry_with_validators(digest), tmp_path, dry_run=False)

    assert success is False
    assert detail != NOT_MODIFIED_DETAIL
    assert not target.exists()
    error.fp.read.assert_not_called()


def test_sync_deleted_local_fetches_unconditionally_and_restores(monkeypatch, tmp_path):
    """A missing local file is fetched in full and written back."""
    target, digest = _write_doc(tmp_path)
    target.unlink()
    html = _html_paragraph("Restored from upstream.")
    entry = _entry_with_validators(digest)

    def responder(req: urllib.request.Request) -> _UrlResponse:
        _assert_unconditional(req)
        return _UrlResponse(
            html.encode("utf-8"),
            {"ETag": '"v2"', "Last-Modified": "Thu, 01 Oct 2026 00:00:00 GMT"},
        )

    _install_urlopen(monkeypatch, responder)
    success, _detail = sync_source(entry, tmp_path, dry_run=False)

    assert success is True
    assert target.is_file()
    assert "Restored from upstream." in target.read_text(encoding="utf-8")
    assert entry.content_hash != digest
    assert entry.etag == '"v2"'


def test_unreadable_local_file_checks_unconditionally_and_does_not_match(monkeypatch, tmp_path):
    """One undecodable file is an absent baseline, not an abort, and not MATCH."""
    target, digest = _write_doc(tmp_path)
    target.write_bytes(b"\xff\xfe not utf-8")
    html = _html_paragraph("Kept body.")
    _title, body = extract_content(html, "html_selector", "#body")
    assert compute_content_hash(extract_substantive_body(body)) == digest

    def responder(req: urllib.request.Request) -> _UrlResponse:
        _assert_unconditional(req)
        return _UrlResponse(html.encode("utf-8"), {"ETag": '"v2"'})

    _install_urlopen(monkeypatch, responder)
    result = check_source_drift(_entry_with_validators(digest), tmp_path)

    assert result.status == "DRIFT_DETECTED"
    assert result.status != "MATCH"
    assert result.local_hash is None
    assert result.upstream_hash == digest
    assert result.error == "Local file is absent"


@patch("scripts.sync_sources.generate_manifest")
def test_unreadable_local_file_does_not_abort_sync(mock_manifest, monkeypatch, tmp_path):
    """--sync still restores an unreadable file and continues to the next source."""
    bad, bad_digest = _write_doc(tmp_path)
    bad.write_bytes(b"\xff\xfe")
    good = tmp_path / "app" / "data" / "good.md"
    good_body = "# Good\n\nStable body.\n"
    good.write_text(good_body, encoding="utf-8")
    good_digest = _digest(good_body)
    good_fingerprint = extraction_fingerprint(SourceEntry(
        path="app/data/good.md",
        url="https://example.com/good",
        type="html_selector",
        selector="#body",
    ))
    config = tmp_path / "sources.yaml"
    config.write_text(
        "# Registry header\n"
        "\n"
        "sources:\n"
        "  - path: app/data/doc.md\n"
        "    url: https://example.com/bad\n"
        "    type: html_selector\n"
        "    selector: '#body'\n"
        "    category: primary\n"
        f"    content_hash: {bad_digest}\n"
        "    etag: '\"stale\"'\n"
        "    upstream_last_modified: 'Wed, 21 Aug 2024 12:00:00 GMT'\n"
        "  - path: app/data/good.md\n"
        "    url: https://example.com/good\n"
        "    type: html_selector\n"
        "    selector: '#body'\n"
        "    category: primary\n"
        f"    content_hash: {good_digest}\n"
        "    etag: '\"good\"'\n"
        "    upstream_last_modified: 'Wed, 21 Aug 2024 12:00:00 GMT'\n"
        f"    extraction_fingerprint: {good_fingerprint}\n",
        encoding="utf-8",
    )
    seen: list[str] = []

    def responder(req: urllib.request.Request) -> object:
        seen.append(req.full_url)
        sent = _request_headers(req)
        if req.full_url.endswith("/bad"):
            assert "if-none-match" not in sent
            return _UrlResponse(
                _html_paragraph("Restored from upstream.").encode("utf-8"),
                {"ETag": '"v2"', "Last-Modified": "Thu, 01 Oct 2026 00:00:00 GMT"},
            )
        assert sent["if-none-match"] == '"good"'
        raise _http_304(req.full_url)

    _install_urlopen(monkeypatch, responder)
    monkeypatch.setattr(sync_sources, "_REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        "sys.argv",
        ["sync_sources.py", "--config", str(config), "--sync"],
    )

    assert sync_sources.main() == 0
    mock_manifest.assert_called_once()
    assert seen == ["https://example.com/bad", "https://example.com/good"]
    assert "Restored from upstream." in bad.read_text(encoding="utf-8")
    assert good.read_text(encoding="utf-8") == good_body


def test_sync_unsolicited_304_is_not_success(monkeypatch, tmp_path):
    """--sync exits non-zero when an unconditional request receives 304."""
    _target, digest = _write_doc(tmp_path)
    config = tmp_path / "sources.yaml"
    config.write_text(
        "# Registry header\n"
        "\n"
        "sources:\n"
        "  - path: app/data/doc.md\n"
        "    url: https://example.com/doc\n"
        "    type: html_selector\n"
        "    selector: '#body'\n"
        "    category: primary\n"
        f"    content_hash: {digest}\n"
        "    last_synced: '2000-01-01'\n",
        encoding="utf-8",
    )
    error = _http_304()

    def urlopen(req: urllib.request.Request, timeout: int) -> object:
        sent = {key.lower(): value for key, value in req.header_items()}
        assert "if-none-match" not in sent
        assert "if-modified-since" not in sent
        assert timeout == 15
        raise error

    monkeypatch.setattr(sync_sources.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(sync_sources, "_REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        "sys.argv",
        ["sync_sources.py", "--config", str(config), "--sync"],
    )

    assert sync_sources.main() != 0
    error.fp.read.assert_not_called()


_TWO_SECTIONS = (
    "<html><body>"
    "<div id='body'><p>Kept body.</p></div>"
    "<div id='other'><p>Other section.</p></div>"
    "</body></html>"
)


def _stored_fingerprint_for(selector: str) -> str:
    return extraction_fingerprint(SourceEntry(
        path="app/data/doc.md",
        url="https://example.com/doc",
        type="html_selector",
        selector=selector,
    ))


def test_check_changed_url_refetches_and_does_not_match(monkeypatch, tmp_path):
    """A new URL must not reuse validators stored for the previous page."""
    _target, digest = _write_doc(tmp_path)
    entry = _entry_with_validators(digest)
    entry.extraction_fingerprint = extraction_fingerprint(entry)
    entry.url = "https://example.com/moved"
    html = _html_paragraph("Kept body.")
    _title, body = extract_content(html, "html_selector", "#body")
    assert compute_content_hash(extract_substantive_body(body)) == digest

    def responder(req: urllib.request.Request) -> _UrlResponse:
        _assert_unconditional(req)
        assert req.full_url == "https://example.com/moved"
        return _UrlResponse(html.encode("utf-8"), {"ETag": '"v2"'})

    _install_urlopen(monkeypatch, responder)
    result = check_source_drift(entry, tmp_path)

    assert result.status == "DRIFT_DETECTED"
    assert result.status != "MATCH"
    assert result.error == "Extraction inputs changed"
    assert result.upstream_hash == digest


def test_sync_changed_url_does_not_skip_the_write(monkeypatch, tmp_path):
    """--sync rewrites the file when the registry URL no longer matches the fingerprint."""
    target, digest = _write_doc(tmp_path)
    before = target.read_text(encoding="utf-8")
    entry = _entry_with_validators(digest)
    entry.extraction_fingerprint = extraction_fingerprint(entry)
    entry.url = "https://example.com/moved"

    def responder(req: urllib.request.Request) -> _UrlResponse:
        _assert_unconditional(req)
        assert req.full_url == "https://example.com/moved"
        return _UrlResponse(
            _html_paragraph("Kept body.").encode("utf-8"),
            {"ETag": '"v2"', "Last-Modified": "Thu, 01 Oct 2026 00:00:00 GMT"},
        )

    _install_urlopen(monkeypatch, responder)
    success, detail = sync_source(entry, tmp_path, dry_run=False)

    assert success is True
    assert detail != NOT_MODIFIED_DETAIL
    written = target.read_text(encoding="utf-8")
    assert written != before
    assert "https://example.com/moved" in written
    assert entry.extraction_fingerprint == extraction_fingerprint(entry)
    assert entry.etag == '"v2"'


def test_check_changed_selector_refetches_and_does_not_match(monkeypatch, tmp_path):
    """A new selector must not reuse validators stored for the old extraction."""
    _target, digest = _write_doc(tmp_path)
    entry = _entry_with_validators(digest)
    entry.selector = "#other"
    entry.extraction_fingerprint = _stored_fingerprint_for("#body")

    def responder(req: urllib.request.Request) -> _UrlResponse:
        _assert_unconditional(req)
        return _UrlResponse(_TWO_SECTIONS.encode("utf-8"), {"ETag": '"v2"'})

    _install_urlopen(monkeypatch, responder)
    result = check_source_drift(entry, tmp_path)

    assert result.status == "DRIFT_DETECTED"
    assert result.status != "MATCH"
    assert result.upstream_hash != digest


def test_sync_changed_selector_does_not_skip_the_write(monkeypatch, tmp_path):
    """--sync re-extracts when the selector no longer matches the stored fingerprint."""
    target, digest = _write_doc(tmp_path)
    entry = _entry_with_validators(digest)
    entry.selector = "#other"
    entry.extraction_fingerprint = _stored_fingerprint_for("#body")

    def responder(req: urllib.request.Request) -> _UrlResponse:
        _assert_unconditional(req)
        return _UrlResponse(_TWO_SECTIONS.encode("utf-8"), {"ETag": '"v2"'})

    _install_urlopen(monkeypatch, responder)
    success, detail = sync_source(entry, tmp_path, dry_run=False)

    assert success is True
    assert detail != NOT_MODIFIED_DETAIL
    written = target.read_text(encoding="utf-8")
    assert "Other section." in written
    assert "Kept body." not in written
    assert entry.extraction_fingerprint == extraction_fingerprint(entry)


def test_check_extractor_version_change_does_not_match(monkeypatch, tmp_path):
    """A bumped extractor version is a full fetch, not a MATCH, even if the text matches."""
    _target, digest = _write_doc(tmp_path)
    entry = _entry_with_validators(digest)
    entry.extraction_fingerprint = extraction_fingerprint(entry)
    monkeypatch.setattr(sync_sources, "EXTRACTOR_VERSION", "2")
    html = _html_paragraph("Kept body.")
    _title, body = extract_content(html, "html_selector", "#body")
    assert compute_content_hash(extract_substantive_body(body)) == digest

    def responder(req: urllib.request.Request) -> _UrlResponse:
        _assert_unconditional(req)
        return _UrlResponse(html.encode("utf-8"), {"ETag": '"v2"'})

    _install_urlopen(monkeypatch, responder)
    result = check_source_drift(entry, tmp_path)

    assert result.status == "DRIFT_DETECTED"
    assert result.status != "MATCH"
    assert result.error == "Extraction inputs changed"
    assert result.upstream_hash == digest


def test_sync_extractor_version_change_does_not_skip_the_write(monkeypatch, tmp_path):
    """--sync rewrites the file when EXTRACTOR_VERSION no longer matches the registry."""
    target, digest = _write_doc(tmp_path)
    before = target.read_text(encoding="utf-8")
    entry = _entry_with_validators(digest)
    entry.extraction_fingerprint = extraction_fingerprint(entry)
    monkeypatch.setattr(sync_sources, "EXTRACTOR_VERSION", "2")

    def responder(req: urllib.request.Request) -> _UrlResponse:
        _assert_unconditional(req)
        return _UrlResponse(_html_paragraph("Kept body.").encode("utf-8"), {"ETag": '"v2"'})

    _install_urlopen(monkeypatch, responder)
    success, detail = sync_source(entry, tmp_path, dry_run=False)

    assert success is True
    assert detail != NOT_MODIFIED_DETAIL
    assert target.read_text(encoding="utf-8") != before
    assert "Kept body." in target.read_text(encoding="utf-8")
    assert entry.extraction_fingerprint == extraction_fingerprint(entry)


@patch("scripts.sync_sources.fetch_upstream")
def test_check_source_drift_pdf_match(mock_fetch, tmp_path):
    """PDF source reports MATCH when upstream bytes match local binary hash."""
    pdf_content = b"%PDF-1.4 mock contract binary content"
    pdf_hash = hashlib.sha256(pdf_content).hexdigest()
    doc_path = tmp_path / "app/public/docs/contract.pdf"
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    doc_path.write_bytes(pdf_content)

    entry = SourceEntry(
        path="app/public/docs/contract.pdf",
        url="https://example.com/contract.pdf",
        type="pdf",
        category="primary",
        content_hash=pdf_hash,
    )
    mock_fetch.return_value = (pdf_content, {"last-modified": "Wed, 01 Oct 2026 12:00:00 GMT"})

    res = check_source_drift(entry, tmp_path)
    assert res.status == "MATCH"
    assert res.local_hash == pdf_hash
    assert res.upstream_hash == pdf_hash


@patch("scripts.sync_sources.fetch_upstream")
def test_check_source_drift_pdf_detected(mock_fetch, tmp_path):
    """PDF source reports DRIFT_DETECTED when upstream bytes change."""
    local_content = b"%PDF-1.4 original contract"
    remote_content = b"%PDF-1.4 revised contract with amendments"
    local_hash = hashlib.sha256(local_content).hexdigest()
    remote_hash = hashlib.sha256(remote_content).hexdigest()

    doc_path = tmp_path / "app/public/docs/contract.pdf"
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    doc_path.write_bytes(local_content)

    entry = SourceEntry(
        path="app/public/docs/contract.pdf",
        url="https://example.com/contract.pdf",
        type="pdf",
        category="primary",
        content_hash=local_hash,
    )
    mock_fetch.return_value = (remote_content, {"last-modified": "Thu, 02 Oct 2026 12:00:00 GMT"})

    res = check_source_drift(entry, tmp_path)
    assert res.status == "DRIFT_DETECTED"
    assert res.local_hash == local_hash
    assert res.upstream_hash == remote_hash


@patch("scripts.sync_sources.fetch_upstream")
def test_check_source_drift_pdf_304(mock_fetch, tmp_path):
    """HTTP 304 to a PDF conditional request reports MATCH with zero bytes parsed."""
    pdf_content = b"%PDF-1.4 intact contract"
    pdf_hash = hashlib.sha256(pdf_content).hexdigest()
    doc_path = tmp_path / "app/public/docs/contract.pdf"
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    doc_path.write_bytes(pdf_content)

    entry = SourceEntry(
        path="app/public/docs/contract.pdf",
        url="https://example.com/contract.pdf",
        type="pdf",
        category="primary",
        content_hash=pdf_hash,
        etag='"etag123"',
    )
    entry.extraction_fingerprint = extraction_fingerprint(entry)
    mock_fetch.side_effect = sync_sources.NotModified({"etag": '"etag123"'})

    res = check_source_drift(entry, tmp_path)
    assert res.status == "MATCH"
    assert res.local_hash == pdf_hash


@patch("scripts.sync_sources.fetch_upstream")
def test_sync_source_pdf_writes_binary(mock_fetch, tmp_path):
    """Syncing a PDF source writes raw bytes to target path and records sha256."""
    pdf_bytes = b"%PDF-1.4 downloaded contract bytes"
    expected_hash = hashlib.sha256(pdf_bytes).hexdigest()

    entry = SourceEntry(
        path="app/public/docs/contract.pdf",
        url="https://example.com/contract.pdf",
        type="pdf",
        category="primary",
    )
    mock_fetch.return_value = (pdf_bytes, {"last-modified": "Wed, 01 Oct 2026 12:00:00 GMT", "etag": '"abc"'})

    success, result_hash = sync_source(entry, tmp_path, dry_run=False)
    assert success is True
    assert result_hash == expected_hash

    target = tmp_path / "app/public/docs/contract.pdf"
    assert target.exists()
    assert target.read_bytes() == pdf_bytes
    assert entry.content_hash == expected_hash
    assert entry.etag == '"abc"'


def test_local_substantive_hash_pdf_oserror(monkeypatch, tmp_path):
    """An unreadable local PDF returns None instead of raising OSError."""
    doc_path = tmp_path / "app/public/docs/locked.pdf"
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    doc_path.write_bytes(b"%PDF-1.4 dummy")

    def mock_read_bytes(self):
        raise OSError("Permission denied")

    monkeypatch.setattr(Path, "read_bytes", mock_read_bytes)
    entry = SourceEntry(
        path="app/public/docs/locked.pdf",
        url="https://example.com/locked.pdf",
        type="pdf",
        category="primary",
    )
    assert _local_substantive_hash(entry, tmp_path) is None


@patch("scripts.sync_sources.fetch_upstream")
def test_check_source_drift_pdf_invalid_payload(mock_fetch, tmp_path):
    """HTML error page returned for a PDF source reports ERROR instead of MATCH/DRIFT."""
    entry = SourceEntry(
        path="app/public/docs/contract.pdf",
        url="https://example.com/contract.pdf",
        type="pdf",
        category="primary",
        content_hash="abc",
    )
    mock_fetch.return_value = (b"<html><body>502 Bad Gateway</body></html>", {})

    res = check_source_drift(entry, tmp_path)
    assert res.status == "ERROR"
    assert "not a valid PDF" in (res.error or "")


@patch("scripts.sync_sources.fetch_upstream")
def test_sync_source_pdf_rejects_invalid_payload(mock_fetch, tmp_path):
    """Syncing a non-PDF payload fails closed without replacing the local file."""
    initial_bytes = b"%PDF-1.4 original valid contract"
    doc_path = tmp_path / "app/public/docs/contract.pdf"
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    doc_path.write_bytes(initial_bytes)

    entry = SourceEntry(
        path="app/public/docs/contract.pdf",
        url="https://example.com/contract.pdf",
        type="pdf",
        category="primary",
        content_hash=hashlib.sha256(initial_bytes).hexdigest(),
    )
    mock_fetch.return_value = (b"<!DOCTYPE html><html>404 Not Found</html>", {})

    success, error = sync_source(entry, tmp_path, dry_run=False)
    assert success is False
    assert "not a valid PDF" in error
    assert doc_path.read_bytes() == initial_bytes

