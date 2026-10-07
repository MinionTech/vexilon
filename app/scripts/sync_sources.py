#!/usr/bin/env python3
"""
Standard Ingestion, Provenance Tracking, and Drift Detection Utility
--------------------------------------------------------------------
Automates ingestion, provenance tracking, and upstream drift detection
for web-sourced knowledge base documents in app/data/.

Issue #695: Automate web-sourced document ingestion, provenance tracking,
and drift detection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sys
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

# Ensure project root is in sys.path
_SCRIPT_DIR = Path(__file__).resolve().parent
_APP_ROOT = _SCRIPT_DIR.parent
_REPO_ROOT = _APP_ROOT.parent

if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
if str(_APP_ROOT) not in sys.path:
    sys.path.insert(0, str(_APP_ROOT))

import yaml
from yaml.constructor import ConstructorError
from yaml.nodes import MappingNode


from generate_cache_manifest import generate_manifest

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("sync_sources")

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 VexilonDocIngestion/1.0"

# Bump when HTMLContentExtractor, clean_bclaws_content, or _strip_bclaws_chrome
# changes the Markdown those functions emit. A new value forces a full fetch.
EXTRACTOR_VERSION = "1"


@dataclass
class SourceEntry:
    path: str
    url: str
    type: str
    selector: str | None = None
    # When set, keep only this part (from its heading through the line before the next part).
    part: str | None = None
    category: str = "general"
    content_hash: str | None = None
    last_synced: str | None = None
    # HTTP validators from the last --sync. Absent until a sync stores them.
    etag: str | None = None
    upstream_last_modified: str | None = None
    # Fingerprint of type, selector, part, url, and EXTRACTOR_VERSION at last --sync.
    extraction_fingerprint: str | None = None


@dataclass
class DriftResult:
    path: str
    url: str
    status: str  # "MATCH", "DRIFT_DETECTED", "ERROR", "UNTRACKED"
    local_hash: str | None = None
    upstream_hash: str | None = None
    upstream_last_modified: str | None = None
    error: str | None = None


class SelectorNotFoundError(Exception):
    """The configured container selector is absent from the upstream HTML.

    Falling through to the whole page would hash chrome (toolbars, copyright
    banners) and let ``--sync`` overwrite a committed legal file with that chrome.
    """

    def __init__(self, selector: str | None = None) -> None:
        self.selector = selector or ""
        if self.selector:
            super().__init__(f"Selector not found in upstream HTML: {self.selector}")
        else:
            super().__init__("Source is missing its selector")


class PartNotFoundError(Exception):
    """The registry named a part whose heading is not in the extracted page.

    Writing the rest of the page would put a sibling part in this file, and the
    bot would cite the wrong document. Fail closed and leave the file untouched.
    """

    def __init__(self, part: str) -> None:
        self.part = part
        super().__init__(f"Part heading not found in upstream document: Part {part}")


class RegistryError(Exception):
    """sources.yaml breaks a cross-entry constraint."""


# sync_source returns this detail on 304 so --sync does not rewrite the registry.
NOT_MODIFIED_DETAIL = "304 Not Modified"


class NotModified(Exception):
    """Upstream returned HTTP 304. The response body was not read or parsed."""

    def __init__(self, headers: dict[str, str]) -> None:
        self.headers = headers
        super().__init__(NOT_MODIFIED_DETAIL)


class _StrictSafeLoader(yaml.SafeLoader):
    """PyYAML SafeLoader that raises ConstructorError on duplicate mapping keys.

    PyYAML's default SafeLoader silently overwrites duplicate keys with the last
    value.  A duplicate ``path`` or ``url`` in sources.yaml would silently
    redirect a scheduled drift check; a duplicate ``content_hash`` would corrupt
    the drift baseline.  This loader treats duplicates as hard errors.
    Merge keys (<<) are preserved and never flagged.
    """

    def construct_mapping(self, node: MappingNode, deep: bool = False) -> dict:  # type: ignore[override]
        if isinstance(node, MappingNode):
            seen: set[object] = set()
            for key_node, _ in node.value:
                if key_node.tag == "tag:yaml.org,2002:merge":
                    continue
                key = self.construct_object(key_node, deep=deep)
                try:
                    is_dup = key in seen
                except TypeError:
                    continue
                if is_dup:
                    raise ConstructorError(
                        "while constructing a mapping",
                        node.start_mark,
                        f"found duplicate key ({key!r})",
                        key_node.start_mark,
                    )
                seen.add(key)
        return super().construct_mapping(node, deep=deep)


class SimpleYamlLoader:
    """Thin YAML wrapper using PyYAML with strict duplicate-key detection."""

    @staticmethod
    def load(stream_or_str: str) -> dict[str, Any]:
        return yaml.load(stream_or_str, Loader=_StrictSafeLoader) or {}  # noqa: S506 — loader is explicitly strict

    @staticmethod
    def dump(data: dict[str, Any]) -> str:
        return yaml.safe_dump(data, sort_keys=False)


class _OpeningMarker(str):
    """An emphasis marker that opens a span, so cell text can keep whitespace outside it."""


class _ClosingMarker(str):
    """An emphasis marker that closes a span."""


class _LineBreak(str):
    """A ``<br>``. Inside a table cell it separates the cell's segments."""


# A cell is (segments split at <br>, colspan, rowspan).
_Cell = tuple[list[str], int, int]


@dataclass
class _TableState:
    """One open ``<table>``. Only data tables collect rows; layout tables stay inline text."""

    is_data: bool
    outer_tokens: list[str]
    outside_tokens: list[str]
    rows: list[list[_Cell]]
    row: list[_Cell] | None = None
    cell_span: tuple[int, int] | None = None
    cell_tokens: list[str] | None = None


def _lead_space_outside_openers(tokens: list[str]) -> list[str]:
    """Move whitespace that follows an opening marker to before it.

    ``<strong> Description</strong>`` would give ``** Description**``; Markdown
    does not open emphasis before whitespace, so it must be `` **Description**``.
    """
    out: list[str] = []
    waiting: list[str] = []
    for token in tokens:
        if isinstance(token, _OpeningMarker):
            waiting.append(token)
            continue
        if waiting:
            text = token.lstrip()
            out.append(token[: len(token) - len(text)])
            if not text:
                continue
            out.extend(waiting)
            waiting = []
            token = text
        out.append(token)
    out.extend(waiting)
    return out


def _cell_segments(tokens: list[str]) -> list[str]:
    """Split a cell's tokens at each ``<br>`` into one-line Markdown segments.

    Emphasis still open at a break is closed there and reopened in the next
    segment, so each segment is valid on its own. Empty segments are dropped.
    """
    parts: list[list[str]] = []
    current: list[str] = []
    open_markers: list[str] = []
    for token in tokens:
        if isinstance(token, _LineBreak):
            while current and not current[-1].strip():
                current.pop()
            if current and not isinstance(current[-1], (_OpeningMarker, _ClosingMarker)):
                current[-1] = current[-1].rstrip()
            current.extend(_ClosingMarker(m) for m in reversed(open_markers))
            parts.append(current)
            current = [_OpeningMarker(m) for m in open_markers]
            continue
        if isinstance(token, _OpeningMarker):
            open_markers.append(str(token))
        elif isinstance(token, _ClosingMarker) and str(token) in open_markers:
            del open_markers[len(open_markers) - 1 - open_markers[::-1].index(str(token))]
        current.append(token)
    parts.append(current)

    segments = []
    for part in parts:
        text = re.sub(r"\s+", " ", "".join(_lead_space_outside_openers(part))).strip()
        if text.replace("*", "").strip():
            segments.append(text.replace("|", r"\|"))
    return segments


def _span(value: str | None, limit: int) -> int:
    """Parse a colspan/rowspan value; missing, zero, or malformed values count as 1."""
    try:
        return min(max(int((value or "").strip()), 1), limit)
    except ValueError:
        return 1


def _expand_br_rows(
    cells: list[_Cell], covered: bool, is_header: bool
) -> list[list[tuple[str, int, int]]]:
    """Turn one ``<tr>`` into Markdown rows.

    Sources pack several rows into one ``<tr>`` with ``<br>`` between values
    ("1 whistle<br>2 whistles" next to "STOP<br>BACK UP"). When every non-empty
    cell has the same number of segments n > 1, emit n rows and pair segment i
    across cells. Otherwise keep one row and join each cell's segments with
    ``<br>``.

    Not split: the header row of a multi-row table ("Column 1<br>Matter" is one
    label; splitting would turn the label into a data row), and rows that start
    or sit under a rowspan, because the span would then cover the wrong rows.
    """
    counts = {len(segments) for segments, _, _ in cells if segments}
    lines = counts.pop() if len(counts) == 1 else 1
    splittable = not is_header and not covered and all(rowspan == 1 for _, _, rowspan in cells)
    if lines > 1 and splittable:
        return [
            [(segments[i] if segments else "", colspan, 1) for segments, colspan, _ in cells]
            for i in range(lines)
        ]
    return [[("<br>".join(segments), colspan, rowspan) for segments, colspan, rowspan in cells]]


def _render_markdown_table(rows: list[list[_Cell]]) -> str:
    """Render collected rows as a Markdown pipe table. The first row is the header.

    Spans: a spanned value is written once, in its first (top-left) cell; the
    other cells the span covers are left empty. Later values stay in their own
    columns, and no text is duplicated.
    """
    grid: list[list[str]] = []
    # column -> rows still covered by a rowspan from an earlier row
    pending: dict[int, int] = {}
    for index, row in enumerate(rows):
        covered = any(n > 0 for n in pending.values())
        is_header = index == 0 and len(rows) > 1
        for cells in _expand_br_rows(row, covered=covered, is_header=is_header):
            grid.append(_layout_row(cells, pending))

    if not any(cell for row in grid for cell in row):
        return ""
    width = max(len(row) for row in grid)
    lines = []
    for index, row in enumerate(grid):
        padded = row + [""] * (width - len(row))
        lines.append("| " + " | ".join(padded) + " |")
        if index == 0:
            lines.append("| " + " | ".join(["---"] * width) + " |")
    return "\n".join(lines)


def _layout_row(cells: list[tuple[str, int, int]], pending: dict[int, int]) -> list[str]:
    """Place one row's cells into columns, skipping columns covered by earlier rowspans."""
    out: list[str] = []
    col = 0

    def skip_covered() -> None:
        nonlocal col
        while pending.get(col, 0) > 0:
            pending[col] -= 1
            out.append("")
            col += 1

    for text, colspan, rowspan in cells:
        skip_covered()
        for offset in range(colspan):
            out.append(text if offset == 0 else "")
            if rowspan > 1:
                pending[col + offset] = rowspan - 1
        col += colspan
    last_covered = max((c for c, n in pending.items() if n > 0 and c >= col), default=-1)
    while col <= last_covered:
        if pending.get(col, 0) > 0:
            pending[col] -= 1
        out.append("")
        col += 1
    return out


class HTMLContentExtractor(HTMLParser):
    """
    Robust HTML to Markdown content extractor.
    Strips scripts, styles, layout navigation, and extracts targeted containers.

    Tables: a ``<table>`` whose ``border`` attribute is present and not ``0`` is
    a data table and becomes a Markdown pipe table. BC Laws marks every grid
    table this way; its chrome, deposit line, formulas, "where" definitions, and
    contents list use ``border="0"`` or none. Those layout tables keep the
    flowing-text output (one paragraph per row). A table nested inside a data
    table is flattened into its enclosing cell, because Markdown has no nested tables.
    ``<br>`` inside a data cell either splits the row (see ``_expand_br_rows``)
    or stays as ``<br>`` in the cell; other whitespace in a cell becomes one space.
    """

    IGNORABLE_TAGS = {
        "script", "style", "nav", "header", "footer", "aside", "svg",
        "noscript", "iframe", "button", "form"
    }

    BLOCK_TAGS = {"p", "div", "section", "article", "blockquote", "tr"}

    VOID_TAGS = {
        "area", "base", "br", "col", "embed", "hr", "img", "input",
        "link", "meta", "param", "source", "track", "wbr",
    }


    def __init__(self, target_selector: str | None = None):
        super().__init__()
        self.target_selector = target_selector
        # A comma-separated selector list extracts every listed container in document order.
        self.selectors = [s.strip() for s in (target_selector or "").split(",") if s.strip()]
        self.found_selectors: set[str] = set()
        self.inside_target = target_selector is None
        self.selector_depth = 0
        self.tag_stack: list[str] = []
        self.ignore_depth = 0

        self.extracted_title: str | None = None
        self.in_title = False

        self.tokens: list[str] = []
        self.current_link: str | None = None
        self.heading_level: int = 0
        self.is_bold = False
        self.is_italic = False
        self.link_text_tokens: list[str] = []
        self.tables: list[_TableState] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag_lower = tag.lower()
        attr_dict = {k.lower(): (v or "") for k, v in attrs}

        # Track title tag
        if tag_lower == "title":
            self.in_title = True
            return

        if tag_lower in self.IGNORABLE_TAGS:
            self.ignore_depth += 1
            return

        if self.ignore_depth > 0:
            return

        # A listed container nested inside an active one is recorded as found but
        # not re-entered, so its content is emitted once as part of the outer container.
        if self.target_selector:
            matched = self._matches_selector(tag_lower, attr_dict)
            if matched:
                self.found_selectors.add(matched)
                if not self.inside_target:
                    self.inside_target = True
                    self.selector_depth = 1
                    return

        if self.inside_target and self.target_selector and self.selector_depth > 0:
            if tag_lower not in self.VOID_TAGS:
                self.selector_depth += 1

        if not self.inside_target:
            return

        if tag_lower == "table":
            self._open_table(attr_dict)
        elif tag_lower == "tr" and self._in_data_table():
            self._open_row()
        elif tag_lower in ("td", "th") and self._in_data_table():
            self._open_cell(attr_dict)
        # Heading tags
        elif re.match(r"^h[1-6]$", tag_lower):
            self.heading_level = int(tag_lower[1])
            self.tokens.append(f"\n\n{'#' * self.heading_level} ")
        elif tag_lower in self.BLOCK_TAGS:
            self.tokens.append("\n\n")
        elif tag_lower == "br":
            self.tokens.append(_LineBreak("\n"))
        elif tag_lower == "li":
            self.tokens.append("\n- ")
        elif tag_lower in ("strong", "b"):
            self.is_bold = True
            if not self.current_link:
                self.tokens.append(_OpeningMarker("**"))
        elif tag_lower in ("em", "i"):
            self.is_italic = True
            if not self.current_link:
                self.tokens.append(_OpeningMarker("*"))
        elif tag_lower == "a":
            href = attr_dict.get("href", "")
            if href and not href.startswith("javascript:"):
                self.current_link = href
        elif tag_lower in ("td", "th"):
            # Table cells are separate blocks. Without a separator their text
            # glues together ("Workers' Compensation BoardDeposited").
            self._append_boundary_space()

        if tag_lower not in self.VOID_TAGS:
            self.tag_stack.append(tag_lower)

    def _append_boundary_space(self) -> None:
        if not self.tokens or self.tokens[-1].endswith((" ", "\n")):
            return
        self.tokens.append(" ")

    def _append_closing_marker(self, marker: str) -> None:
        """Close emphasis; in a table cell, move trailing whitespace outside the marker.

        A cell is joined onto one line, so ``<strong>Label:<br></strong>Text``
        would become ``**Label: **Text``. Markdown does not close emphasis after
        whitespace, so the marker must come first: ``**Label:** Text``.
        """
        if not any(t.cell_tokens is self.tokens for t in self.tables):
            self.tokens.append(_ClosingMarker(marker))
            return
        # Popped last-first; line-break tokens are kept whole so cells still split on them.
        trailing: list[str] = []
        while self.tokens and not self.tokens[-1].strip():
            trailing.append(self.tokens.pop())
        if self.tokens and self.tokens[-1] != self.tokens[-1].rstrip():
            kept = self.tokens[-1].rstrip()
            trailing.append(self.tokens[-1][len(kept):])
            self.tokens[-1] = kept
        self.tokens.append(_ClosingMarker(marker))
        self.tokens.extend(reversed(trailing))

    def _in_data_table(self) -> bool:
        return bool(self.tables) and self.tables[-1].is_data

    def _open_table(self, attrs: dict[str, str]) -> None:
        border = attrs.get("border")
        is_data = (
            border is not None
            and border.strip() != "0"
            and not any(t.is_data for t in self.tables)
        )
        state = _TableState(is_data=is_data, outer_tokens=self.tokens, outside_tokens=[], rows=[])
        self.tables.append(state)
        if is_data:
            self.tokens = state.outside_tokens

    def _open_row(self) -> None:
        self._close_row()
        self.tables[-1].row = []

    def _open_cell(self, attrs: dict[str, str]) -> None:
        state = self.tables[-1]
        self._close_cell()
        if state.row is None:
            state.row = []
        # HTML caps colspan at 1000 and rowspan at 65534.
        state.cell_span = (_span(attrs.get("colspan"), 1000), _span(attrs.get("rowspan"), 65534))
        state.cell_tokens = []
        self.tokens = state.cell_tokens

    def _close_cell(self) -> None:
        state = self.tables[-1]
        if state.cell_tokens is None or state.cell_span is None or state.row is None:
            return
        state.row.append((_cell_segments(state.cell_tokens), *state.cell_span))
        state.cell_tokens = None
        state.cell_span = None
        self.tokens = state.outside_tokens

    def _close_row(self) -> None:
        state = self.tables[-1]
        self._close_cell()
        if state.row:
            state.rows.append(state.row)
        state.row = None

    def _close_table(self) -> None:
        if not self.tables:
            return
        if not self.tables[-1].is_data:
            self.tables.pop()
            return
        self._close_row()
        state = self.tables.pop()
        self.tokens = state.outer_tokens
        outside = "".join(state.outside_tokens).strip()
        if outside:
            self.tokens.extend(["\n\n", outside, "\n\n"])
        rendered = _render_markdown_table(state.rows)
        if rendered:
            self.tokens.extend(["\n\n", rendered, "\n\n"])

    def _close_open_tables(self) -> None:
        while self.tables:
            self._close_table()


    def handle_endtag(self, tag: str) -> None:
        tag_lower = tag.lower()

        if tag_lower == "title":
            self.in_title = False
            return

        if tag_lower in self.IGNORABLE_TAGS:
            if self.ignore_depth > 0:
                self.ignore_depth -= 1
            return

        if self.ignore_depth > 0:
            return

        if tag_lower in self.VOID_TAGS:
            return

        if self.inside_target and self.target_selector and self.selector_depth > 0:
            self.selector_depth -= 1
            if self.selector_depth == 0:
                self._close_open_tables()
                self.inside_target = False
                return


        if not self.inside_target:
            return

        if tag_lower == "table":
            self._close_table()
        elif tag_lower == "tr" and self._in_data_table():
            self._close_row()
        elif tag_lower in ("td", "th") and self._in_data_table():
            self._close_cell()
        elif re.match(r"^h[1-6]$", tag_lower):
            self.heading_level = 0
            self.tokens.append("\n\n")
        elif tag_lower in self.BLOCK_TAGS:
            self.tokens.append("\n\n")
        elif tag_lower in ("strong", "b"):
            self.is_bold = False
            if not self.current_link:
                self._append_closing_marker("**")
        elif tag_lower in ("em", "i"):
            self.is_italic = False
            if not self.current_link:
                self._append_closing_marker("*")
        elif tag_lower == "a":
            if self.current_link and self.link_text_tokens:
                anchor_text = "".join(self.link_text_tokens).strip()
                if anchor_text:
                    self.tokens.append(f"[{anchor_text}]({self.current_link})")
            self.current_link = None
            self.link_text_tokens = []

        if self.tag_stack and self.tag_stack[-1] == tag_lower:
            self.tag_stack.pop()

    def handle_data(self, data: str) -> None:
        if self.in_title and not self.extracted_title:
            self.extracted_title = data.strip()
            return

        if self.ignore_depth > 0 or not self.inside_target:
            return

        # A newline inside one element ("CHAPTER\\n1") is a space, not a new paragraph.
        text = re.sub(r"\s+", " ", data)
        if not text:
            return

        if self.current_link:
            self.link_text_tokens.append(text)
            return

        self.tokens.append(text)

    @property
    def selector_found(self) -> bool:
        return all(s in self.found_selectors for s in self.selectors)

    @property
    def missing_selectors(self) -> list[str]:
        return [s for s in self.selectors if s not in self.found_selectors]

    def _matches_selector(self, tag: str, attrs: dict[str, str]) -> str | None:
        """Return the listed selector this element matches, or None."""
        for selector in self.selectors:
            if selector.startswith("#"):
                if attrs.get("id", "") == selector[1:]:
                    return selector
            elif selector.startswith("."):
                if selector[1:] in attrs.get("class", "").split():
                    return selector
            elif tag == selector.lower():
                return selector
        return None

    def get_markdown(self) -> str:
        self._close_open_tables()
        raw_text = "".join(self.tokens)
        # Clean extra spaces & lines
        lines = []
        for line in raw_text.splitlines():
            cleaned_line = re.sub(r"[ \t]+", " ", line).strip()
            lines.append(cleaned_line)
        cleaned = "\n".join(lines)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned.strip()


def _require_selector(extractor: HTMLContentExtractor) -> None:
    """Refuse a whole-page extract when the configured container is missing."""
    if extractor.target_selector and not extractor.selector_found:
        raise SelectorNotFoundError(", ".join(extractor.missing_selectors))


def clean_bclaws_content(raw_html: str, selector: str | None, part: str | None = None) -> str:
    """Specialized cleaner for BC Laws statutory documents.

    ``selector`` is the registry value already chosen for this entry. There is
    no default container: a null selector must not fall through to the whole page.
    """
    if not selector:
        raise SelectorNotFoundError()
    extractor = HTMLContentExtractor(target_selector=selector)
    extractor.feed(raw_html)
    _require_selector(extractor)
    content = extractor.get_markdown()

    # Strip bclaws URL watermarks and boilerplate footers
    content = re.sub(r'https?://www\.bclaws\.gov\.bc\.ca/[^\s)\]>"]+', "", content)
    content = re.sub(r"function\s+launchNewWindow[\s\S]*?\}", "", content)
    content = re.sub(r"window\.onload\s*=[\s\S]*?\}", "", content)
    content = _strip_bclaws_chrome(content)
    content = re.sub(r"\n{3,}", "\n\n", content)
    content = content.strip()
    if part:
        content = slice_bclaws_part(content, part)
    return content


_KING_PRINTER_RE = re.compile(
    r"Copyright © King's Printer,\s*Victoria,\s*British\s*Columbia,\s*Canada",
    re.IGNORECASE,
)
_LICENCE_DISCLAIMER_RE = re.compile(
    r"\[(?:Licence|Disclaimer)\]\([^)]*(?:Licence|Disclaimer)\.html\)",
    re.IGNORECASE,
)
_VIEW_COMPLETE_RE = re.compile(r"\[View Complete [^\]]+\]\([^)]+\)")
_POINT_IN_TIME_RE = re.compile(r"\[Link to Point in Time\]\([^)]+\)")
_CONSOLIDATED_PDF_RE = re.compile(
    r"(?m)^.*\[Link to consolidated regulation \(PDF\)\]\([^)]+\).*$"
)
# Foot nav: "[Contents](...) | [Parts 1 to 3](...) | ...", optional bold or heading marks.
_NAV_RAIL_RE = re.compile(
    r"(?m)^[ \t]*(?:#{1,6}[ \t]*)?\**[ \t]*\[Contents\]\([^)]+\)[ \t]*\|.*$"
)
# Opening and closing markers with nothing between them: **** or ** **.
_EMPTY_BOLD_RE = re.compile(r"\*\*(?:\s*\*\*)+")
_EMPTY_HEADING_RE = re.compile(r"(?m)^#{1,6}[ \t*]*$")
# "Part 2 — Application" or a bare "Part 33". Optional markdown heading marks.
_PART_HEADING_RE = re.compile(
    r"^(?:#{1,6}[ \t]+)?Part (?P<num>\d+)(?:[ \t]+[—–-].*)?[ \t]*$"
)


def _drop_empty_bold(match: re.Match[str]) -> str:
    """Remove an empty bold pair without joining the words or blocks around it.

    "** **" is also one bold run closing and the next opening; dropping the
    markers must keep the space. Across a line break ("**Table 3-1**" then
    "**Minimum ...**") the markers close and reopen bold per block, so they stay.
    """
    run = match.group(0)
    if "\n" in run:
        return run
    return run.replace("*", "")


def _strip_bclaws_chrome(content: str) -> str:
    """Drop King's Printer chrome, nav rails, and empty bold markers from statute text."""
    content = _KING_PRINTER_RE.sub("", content)
    content = _LICENCE_DISCLAIMER_RE.sub("", content)
    content = _VIEW_COMPLETE_RE.sub("", content)
    content = _POINT_IN_TIME_RE.sub("", content)
    content = _CONSOLIDATED_PDF_RE.sub("", content)
    content = _NAV_RAIL_RE.sub("", content)
    content = _EMPTY_BOLD_RE.sub(_drop_empty_bold, content)
    content = _EMPTY_HEADING_RE.sub("", content)
    return content


def slice_bclaws_part(content: str, part: str) -> str:
    """Keep the text from this part's heading up to the next part heading.

    BC Laws publishes several parts in one HTML document. Without this cut, every
    file for that URL receives the whole bundle and the bot cites the wrong part.
    """
    number = str(part).strip()
    if not re.fullmatch(r"\d+", number):
        raise PartNotFoundError(part)
    lines = content.splitlines()
    start: int | None = None
    for index, line in enumerate(lines):
        match = _PART_HEADING_RE.match(line)
        if match and match.group("num") == number:
            start = index
            break
    if start is None:
        raise PartNotFoundError(number)
    end = len(lines)
    for index in range(start + 1, len(lines)):
        if _PART_HEADING_RE.match(lines[index]):
            end = index
            break
    sliced = "\n".join(lines[start:end]).strip()
    if not sliced:
        raise PartNotFoundError(number)
    return sliced


def extract_content(
    raw_html: str,
    doc_type: str,
    selector: str | None,
    part: str | None = None,
) -> tuple[str, str]:
    """Extracts (title, substantive_markdown_body) from raw HTML."""
    if doc_type == "bclaws":
        # One resolved selector for the guard and the body. Do not substitute a
        # container, and do not hash the whole page when the registry omits one.
        resolved_selector = selector
        if not resolved_selector:
            raise SelectorNotFoundError()
        extractor = HTMLContentExtractor(target_selector=resolved_selector)
        extractor.feed(raw_html)
        _require_selector(extractor)
        title = extractor.extracted_title or "BC Statute"
        body = clean_bclaws_content(raw_html, selector=resolved_selector, part=part)
    else:
        if part:
            raise PartNotFoundError(part)
        extractor = HTMLContentExtractor(target_selector=selector or "#body")
        extractor.feed(raw_html)
        _require_selector(extractor)
        title = extractor.extracted_title or "Document"
        body = extractor.get_markdown()

    # If title is in the body's first h1, extract it
    h1_match = re.match(r"^#\s+(.+)$", body, re.MULTILINE)
    if h1_match:
        title = h1_match.group(1).strip()

    return title, body


def extract_substantive_body(markdown_text: str) -> str:
    """
    Strips existing provenance headers and metadata from Markdown text,
    returning strictly the substantive body text for drift hashing.
    """
    lines = markdown_text.splitlines()
    body_lines: list[str] = []
    header_passed = False

    for line in lines:
        stripped = line.strip()
        if not header_passed:
            # Skip initial heading, metadata lines, and divider
            if (
                stripped.startswith("# ")
                or stripped.startswith("**Source:")
                or stripped.startswith("> **Source:")
                or stripped.startswith("**Upstream Last Modified:")
                or stripped.startswith("**Official Page Last Updated:")
                or stripped.startswith("**Ingestion Date:")
                or stripped.startswith("> **Format:")
                or re.match(r"^\*Updated: \d", stripped)
                or stripped == "---"
                or not stripped
            ):
                continue
            header_passed = True
        body_lines.append(line)

    substantive = "\n".join(body_lines).strip()
    return substantive


# BC Laws stamps a currency date on its own line. The date moves on a schedule
# that is not a change to the statute, so a date-only bump must not open a
# drift issue. The line stays in the synced markdown; only the hash omits it.
# Optional leading "**" is the markdown bold wrapper on the consolidation stamp.
_BCLAWS_CURRENCY_LINE_RE = re.compile(
    r"^(?:\*\*)?(?:This Act is current to|This regulation is current to|This consolidation is current to)\b"
)


def _omit_bclaws_currency_lines(text: str) -> str:
    """Drop BC Laws currency-stamp lines before the drift hash is computed."""
    return "\n".join(
        line
        for line in text.splitlines()
        if not _BCLAWS_CURRENCY_LINE_RE.match(line.strip())
    )


def compute_content_hash(text: str) -> str:
    """Compute SHA-256 hash of normalized text.

    A line is omitted when, after stripping surrounding whitespace, it is
    optional leading markdown bold plus "This Act is current to",
    "This regulation is current to", or "This consolidation is current to".
    Any other text remains in the hash.
    """
    normalized = re.sub(r"\s+", " ", _omit_bclaws_currency_lines(text)).strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def format_provenance_header(
    title: str,
    url: str,
    upstream_last_modified: str | None = None,
    ingestion_date: str | None = None,
) -> str:
    """Formats canonical provenance metadata header."""
    # BC Laws <title> text can contain a newline (for example "Workers Compensation\\n    Act").
    # A wrapped heading stops extract_substantive_body early, so the registry hash no longer
    # matches the file the sync just wrote.
    title = " ".join(title.split())
    last_mod = upstream_last_modified or "Unknown"
    ingest = ingestion_date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return (
        f"# {title}\n\n"
        f"**Source:** [{title}]({url})  \n"
        f"**Upstream Last Modified:** {last_mod}  \n"
        f"**Ingestion Date:** {ingest}  \n\n"
    )


def _optional_header(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _validator_kwargs(source: SourceEntry) -> dict[str, str]:
    """Conditional-request kwargs for validators stored on the entry."""
    kwargs: dict[str, str] = {}
    if source.etag:
        kwargs["etag"] = source.etag
    if source.upstream_last_modified:
        kwargs["upstream_last_modified"] = source.upstream_last_modified
    return kwargs


def _local_substantive_hash(source: SourceEntry, repo_root: Path) -> str | None:
    """Substantive hash of the local file, or None when it is absent or unreadable.

    An unreadable file is not a baseline. Callers then fetch unconditionally
    instead of aborting the rest of ``--check`` or ``--sync``.
    """
    local_file = repo_root / source.path
    if not local_file.is_file():
        return None
    if source.type == "pdf":
        return hashlib.sha256(local_file.read_bytes()).hexdigest()
    try:
        text = local_file.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    return compute_content_hash(extract_substantive_body(text))


def _baseline_is_intact(source: SourceEntry, local_hash: str | None) -> bool:
    """True when the local file is still the body the stored validators describe."""
    return bool(source.content_hash) and local_hash is not None and local_hash == source.content_hash


def extraction_fingerprint(source: SourceEntry) -> str:
    """Identity of the page and the inputs that turn its HTML into stored Markdown.

    ``url`` is part of the identity. Validators stored for one page must not be
    sent to a different address after the registry URL changes.
    """
    raw = "\n".join((
        EXTRACTOR_VERSION,
        source.type or "",
        source.selector or "",
        source.part or "",
        source.url,
    ))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _extraction_fingerprint_matches(source: SourceEntry) -> bool:
    """True when this sync stored the fingerprint and the inputs are unchanged."""
    stored = source.extraction_fingerprint
    return bool(stored) and stored == extraction_fingerprint(source)


def _validators_for_intact_baseline(source: SourceEntry, local_hash: str | None) -> dict[str, str]:
    """Send stored validators only when a 304 would skip the bytes already validated.

    The local file must still match ``content_hash``, and the extractor inputs
    stored with the validators must still be the ones in effect. A missing or
    different fingerprint means the Markdown was not produced by this extraction.
    """
    if not _baseline_is_intact(source, local_hash):
        return {}
    if not _extraction_fingerprint_matches(source):
        return {}
    return _validator_kwargs(source)


def _store_validators(source: SourceEntry, headers: dict[str, str]) -> None:
    """Replace stored validators with the ones from a 200 response."""
    source.etag = _optional_header(headers.get("etag"))
    source.upstream_last_modified = _optional_header(headers.get("last-modified"))


def _header_map(headers: object) -> dict[str, str]:
    items = getattr(headers, "items", None)
    if not callable(items):
        return {}
    return {str(key).lower(): str(value) for key, value in items()}


def fetch_upstream(
    url: str,
    timeout: int = 15,
    *,
    etag: str | None = None,
    upstream_last_modified: str | None = None,
    is_binary: bool = False,
) -> tuple[str | bytes, dict[str, str]]:
    """Fetch upstream content and response headers.

    Stored validators are sent as ``If-None-Match`` and ``If-Modified-Since``.
    HTTP 304 raises ``NotModified`` and does not read the body only when this
    request sent a validator. A 304 to an unconditional request stays an error.
    """
    req_headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/pdf,application/octet-stream,*/*" if is_binary else "text/html,application/xhtml+xml",
    }
    if etag:
        req_headers["If-None-Match"] = etag
    if upstream_last_modified:
        req_headers["If-Modified-Since"] = upstream_last_modified
    req = urllib.request.Request(url, headers=req_headers)
    try:
        resp_ctx = urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as exc:
        if exc.code != 304 or not (etag or upstream_last_modified):
            raise
        headers = _header_map(exc.headers)
        exc.close()
        raise NotModified(headers) from None
    with resp_ctx as resp:
        headers = _header_map(resp.headers)
        if is_binary:
            content: str | bytes = resp.read()
        else:
            content = resp.read().decode("utf-8", errors="replace")
        return content, headers


def load_registry(config_path: Path) -> list[SourceEntry]:
    """Loads declarative source registry from sources.yaml."""
    if not config_path.exists():
        raise FileNotFoundError(f"Source registry not found: {config_path}")
    raw_data = SimpleYamlLoader.load(config_path.read_text(encoding="utf-8"))
    entries: list[SourceEntry] = []
    for s in raw_data.get("sources", []):
        raw_part = s.get("part")
        if raw_part is None or (isinstance(raw_part, str) and not raw_part.strip()):
            part = None
        else:
            part = str(raw_part).strip()
        entries.append(
            SourceEntry(
                path=s["path"],
                url=s["url"],
                type=s.get("type", "html_selector"),
                selector=s.get("selector"),
                part=part,
                category=s.get("category", "general"),
                content_hash=s.get("content_hash"),
                last_synced=s.get("last_synced"),
                etag=_optional_header(s.get("etag")),
                upstream_last_modified=_optional_header(s.get("upstream_last_modified")),
                extraction_fingerprint=_optional_header(s.get("extraction_fingerprint")),
            )
        )
    _validate_registry(entries)
    return entries


def _validate_registry(entries: list[SourceEntry]) -> None:
    """Reject shared URLs that do not name distinct parts, and shared baselines.

    Two catalogue entries on one BC Laws document must say which part each file
    keeps. A repeated content_hash means two files were given the same body, so
    a drift check cannot tell them apart.
    """
    parts_by_url: dict[str, list[str | None]] = {}
    paths_by_hash: dict[str, list[str]] = {}
    for entry in entries:
        parts_by_url.setdefault(entry.url, []).append(entry.part)
        if entry.content_hash:
            paths_by_hash.setdefault(entry.content_hash, []).append(entry.path)
    for url, parts in parts_by_url.items():
        if len(parts) > 1 and len(set(parts)) != len(parts):
            raise RegistryError(
                f"Entries that share {url} must have different part values, got {parts}"
            )
    for digest, paths in paths_by_hash.items():
        if len(paths) > 1:
            raise RegistryError(
                f"content_hash {digest} is shared by {paths}"
            )


def leading_comment_block(text: str) -> str:
    """Return the leading ``#`` comment block, including blank lines between comments.

    ``yaml.safe_dump`` drops comments. ``--sync`` rewrites sources.yaml, so the
    schema header has to be captured and written back or the first sync deletes it.
    """
    kept: list[str] = []
    started = False
    for line in text.splitlines():
        if line.startswith("#"):
            started = True
            kept.append(line)
            continue
        if started and line.strip() == "":
            kept.append(line)
            continue
        break
    while kept and kept[-1].strip() == "":
        kept.pop()
    if not kept:
        return ""
    return "\n".join(kept) + "\n\n"


def save_registry(config_path: Path, entries: list[SourceEntry]) -> None:
    """Saves updated entries back to sources.yaml, preserving the leading comment block."""
    existing = config_path.read_text(encoding="utf-8") if config_path.exists() else ""
    _validate_registry(entries)
    header = leading_comment_block(existing)
    data = {
        "sources": [
            {key: value for key, value in asdict(entry).items() if value is not None}
            for entry in entries
        ]
    }
    body = SimpleYamlLoader.dump(data)
    if not body.endswith("\n"):
        body += "\n"
    config_path.write_text(header + body, encoding="utf-8")


def check_source_drift(source: SourceEntry, repo_root: Path) -> DriftResult:
    """
    Checks whether the upstream document has drifted from our local copy or registry hash.
    Does NOT modify local files.
    """
    local_hash = _local_substantive_hash(source, repo_root)
    if local_hash is None and source.content_hash is None:
        return DriftResult(
            path=source.path,
            url=source.url,
            status="UNTRACKED",
        )
    # Registry hash stays in the comparison so a missing file can still show
    # upstream drift. It is not a baseline, and it is not the local hash.
    registry_hash = source.content_hash

    try:
        if source.type == "pdf":
            raw_bytes, headers = fetch_upstream(
                source.url,
                **_validators_for_intact_baseline(source, local_hash),
                is_binary=True,
            )
            upstream_hash = hashlib.sha256(
                raw_bytes if isinstance(raw_bytes, bytes) else raw_bytes.encode()
            ).hexdigest()
            upstream_last_modified = headers.get("last-modified")

            if registry_hash and upstream_hash != registry_hash:
                return DriftResult(
                    path=source.path,
                    url=source.url,
                    status="DRIFT_DETECTED",
                    local_hash=local_hash,
                    upstream_hash=upstream_hash,
                    upstream_last_modified=upstream_last_modified,
                    error="Local file is absent" if local_hash is None else None,
                )
            elif local_hash is not None and upstream_hash != local_hash:
                return DriftResult(
                    path=source.path,
                    url=source.url,
                    status="DRIFT_DETECTED",
                    local_hash=local_hash,
                    upstream_hash=upstream_hash,
                    upstream_last_modified=upstream_last_modified,
                )
            if local_hash is None and registry_hash:
                return DriftResult(
                    path=source.path,
                    url=source.url,
                    status="DRIFT_DETECTED",
                    local_hash=None,
                    upstream_hash=upstream_hash,
                    upstream_last_modified=upstream_last_modified,
                    error="Local file is absent",
                )
            return DriftResult(
                path=source.path,
                url=source.url,
                status="MATCH",
                local_hash=local_hash,
                upstream_hash=upstream_hash,
                upstream_last_modified=upstream_last_modified,
            )

        raw_html, headers = fetch_upstream(
            source.url, **_validators_for_intact_baseline(source, local_hash)
        )
        title, upstream_body = extract_content(
            raw_html, source.type, source.selector, source.part
        )
        upstream_substantive = extract_substantive_body(upstream_body)
        if not upstream_substantive:
            return DriftResult(
                path=source.path,
                url=source.url,
                status="ERROR",
                local_hash=local_hash or registry_hash,
                error="Upstream extraction produced no substantive content",
            )
        upstream_hash = compute_content_hash(upstream_substantive)
        upstream_last_modified = headers.get("last-modified")

        if registry_hash and upstream_hash != registry_hash:
            return DriftResult(
                path=source.path,
                url=source.url,
                status="DRIFT_DETECTED",
                local_hash=local_hash,
                upstream_hash=upstream_hash,
                upstream_last_modified=upstream_last_modified,
                error="Local file is absent" if local_hash is None else None,
            )
        elif local_hash is not None and upstream_hash != local_hash:
            return DriftResult(
                path=source.path,
                url=source.url,
                status="DRIFT_DETECTED",
                local_hash=local_hash,
                upstream_hash=upstream_hash,
                upstream_last_modified=upstream_last_modified,
            )
        if local_hash is None and registry_hash:
            return DriftResult(
                path=source.path,
                url=source.url,
                status="DRIFT_DETECTED",
                local_hash=None,
                upstream_hash=upstream_hash,
                upstream_last_modified=upstream_last_modified,
                error="Local file is absent",
            )
        if source.extraction_fingerprint and not _extraction_fingerprint_matches(source):
            return DriftResult(
                path=source.path,
                url=source.url,
                status="DRIFT_DETECTED",
                local_hash=local_hash,
                upstream_hash=upstream_hash,
                upstream_last_modified=upstream_last_modified,
                error="Extraction inputs changed",
            )

        return DriftResult(
            path=source.path,
            url=source.url,
            status="MATCH",
            local_hash=local_hash,
            upstream_hash=upstream_hash,
            upstream_last_modified=upstream_last_modified,
        )

    except NotModified:
        # Validators are sent only when the local baseline is intact, so a 304
        # here is that same body. fetch_upstream does not raise this otherwise.
        return DriftResult(
            path=source.path,
            url=source.url,
            status="MATCH",
            local_hash=local_hash,
            upstream_hash=registry_hash,
            upstream_last_modified=source.upstream_last_modified,
        )
    except (urllib.error.URLError, TimeoutError) as e:
        logger.warning(f"Network error checking {source.url}: {e}")
        return DriftResult(
            path=source.path,
            url=source.url,
            status="ERROR",
            local_hash=local_hash or registry_hash,
            error=f"Network error: {e}",
        )
    except Exception as e:
        logger.error(f"Failed to check {source.url}: {e}", exc_info=True)
        return DriftResult(
            path=source.path,
            url=source.url,
            status="ERROR",
            local_hash=local_hash or registry_hash,
            error=str(e),
        )


def sync_source(
    source: SourceEntry,
    repo_root: Path,
    dry_run: bool = False,
) -> tuple[bool, str]:
    """
    Syncs upstream document to local markdown file and updates metadata.
    """
    target_path = repo_root / source.path
    local_hash = _local_substantive_hash(source, repo_root)

    if source.type == "pdf":
        try:
            try:
                raw_bytes, headers = fetch_upstream(
                    source.url,
                    **_validators_for_intact_baseline(source, local_hash),
                    is_binary=True,
                )
            except NotModified:
                logger.info(f"Not modified {source.path}; skipped download")
                return True, NOT_MODIFIED_DETAIL

            upstream_last_modified = headers.get("last-modified")
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            new_hash = hashlib.sha256(
                raw_bytes if isinstance(raw_bytes, bytes) else raw_bytes.encode()
            ).hexdigest()

            if not dry_run:
                target_path.parent.mkdir(parents=True, exist_ok=True)
                if isinstance(raw_bytes, bytes):
                    target_path.write_bytes(raw_bytes)
                else:
                    target_path.write_text(raw_bytes, encoding="utf-8")
                source.content_hash = new_hash
                source.last_synced = today
                source.extraction_fingerprint = extraction_fingerprint(source)
                _store_validators(source, headers)
                logger.info(f"Updated {source.path} (hash: {new_hash[:8]})")
            else:
                logger.info(f"[DRY-RUN] Would write {source.path} (hash: {new_hash[:8]})")

            return True, new_hash
        except Exception as e:
            logger.error(f"Failed to sync {source.url}: {e}", exc_info=True)
            return False, str(e)

    try:
        try:
            raw_html, headers = fetch_upstream(
                source.url, **_validators_for_intact_baseline(source, local_hash)
            )
        except NotModified:
            # Same gate as --check: this runs only after validators were sent,
            # which requires the local file to still match content_hash.
            logger.info(f"Not modified {source.path}; skipped parse")
            return True, NOT_MODIFIED_DETAIL
        title, body = extract_content(raw_html, source.type, source.selector, source.part)
        upstream_last_modified = headers.get("last-modified")
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        # Strip any existing leading h1 from body if title matches to prevent duplication
        clean_body = body
        if clean_body.startswith(f"# {title}"):
            clean_body = clean_body[len(f"# {title}"):].strip()

        header = format_provenance_header(
            title=title,
            url=source.url,
            upstream_last_modified=upstream_last_modified,
            ingestion_date=today,
        )
        full_content = header + clean_body + "\n"

        substantive = extract_substantive_body(clean_body)
        if not substantive:
            return False, "Upstream extraction produced no substantive content"
        new_hash = compute_content_hash(substantive)

        if not dry_run:
            target_path.parent.mkdir(parents=True, exist_ok=True)
            target_path.write_text(full_content, encoding="utf-8")
            source.content_hash = new_hash
            source.last_synced = today
            source.extraction_fingerprint = extraction_fingerprint(source)
            _store_validators(source, headers)
            logger.info(f"Updated {source.path} (hash: {new_hash[:8]})")
        else:
            logger.info(f"[DRY-RUN] Would write {source.path} (hash: {new_hash[:8]})")

        return True, new_hash

    except Exception as e:
        logger.error(f"Failed to sync {source.url}: {e}", exc_info=True)
        return False, str(e)


def format_drift_alert_markdown(results: list[DriftResult]) -> str:
    """Formats markdown issue body for drifted documents."""
    drifted = [r for r in results if r.status == "DRIFT_DETECTED"]
    errored = [r for r in results if r.status == "ERROR"]
    lines = [
        "## Upstream Document Drift Detected",
        "",
        "The scheduled upstream drift check detected differences between external sources and the local knowledge base under `app/data/`.",
        "",
        "| Document Path | Upstream URL | Local Hash | Upstream Hash | Upstream Last Modified |",
        "| :--- | :--- | :--- | :--- | :--- |",
    ]
    for d in drifted:
        doc_name = d.path.split("/")[-1]
        local_h = (d.local_hash or "")[:8]
        upstream_h = (d.upstream_hash or "")[:8]
        last_mod = d.upstream_last_modified or "Unknown"
        lines.append(f"| `{d.path}` | [{doc_name}]({d.url}) | `{local_h}` | `{upstream_h}` | {last_mod} |")

    if errored:
        lines.extend([
            "",
            "### Sources That Could Not Be Reached",
            "",
            "| Document Path | Upstream URL | Error |",
            "| :--- | :--- | :--- |",
        ])
        for e in errored:
            doc_name = e.path.split("/")[-1]
            error_msg = (e.error or "Unknown error").replace("|", "\\|")
            lines.append(f"| `{e.path}` | [{doc_name}]({e.url}) | {error_msg} |")

    lines.extend([
        "",
        "### Steward Action Required",
        "1. Review the upstream changes at the provided URLs.",
        "2. Run `python app/scripts/sync_sources.py --sync` locally.",
        "3. Inspect the `git diff` to verify legal and substantive integrity.",
        "4. Commit and push the candidate changes in a PR referencing this issue.",
    ])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Web-sourced document ingestion and drift detection.")
    parser.add_argument(
        "--config", "-c",
        type=Path,
        default=_APP_ROOT / "data" / "sources.yaml",
        help="Path to sources.yaml registry",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Run drift detection check without modifying documents",
    )
    parser.add_argument(
        "--sync",
        action="store_true",
        help="Fetch upstream content and sync local files",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate sync without writing files",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output check results in JSON format",
    )
    parser.add_argument(
        "--alert-body",
        type=Path,
        default=None,
        help="Write GitHub issue markdown body to file if drift is detected",
    )
    parser.add_argument(
        "--filter", "-f",
        type=str,
        default=None,
        help="Filter sources by path or url substring",
    )

    args = parser.parse_args()

    if args.check and args.sync:
        parser.error("--check and --sync are mutually exclusive: --check reads only, --sync writes.")

    try:
        return _run(args)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return 2


def _run(args: argparse.Namespace) -> int:
    config_path = args.config.resolve()
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        return 2

    registry = load_registry(config_path)
    entries = registry
    if args.filter:
        entries = [e for e in registry if args.filter in e.path or args.filter in e.url]
        logger.info(f"Filtered to {len(entries)} source(s)")

    if args.check or (not args.sync and not args.dry_run):
        # Run drift check
        results: list[DriftResult] = []
        has_drift = False
        has_error = False

        for entry in entries:
            logger.info(f"Checking {entry.path}...")
            res = check_source_drift(entry, _REPO_ROOT)
            results.append(res)
            if res.status == "DRIFT_DETECTED":
                has_drift = True
            elif res.status == "ERROR":
                has_error = True

        if args.alert_body and has_drift:
            args.alert_body.resolve().write_text(
                format_drift_alert_markdown(results), encoding="utf-8"
            )
            logger.info(f"Wrote drift alert body to {args.alert_body}")

        if args.json:
            print(json.dumps([asdict(r) for r in results], indent=2))
        else:
            print("\n" + "=" * 60)
            print("DRIFT DETECTION REPORT")
            print("=" * 60)
            for r in results:
                if r.status == "MATCH":
                    status_symbol = "✅"
                elif r.status == "DRIFT_DETECTED":
                    status_symbol = "⚠️"
                elif r.status == "UNTRACKED":
                    status_symbol = "ℹ️"
                else:
                    status_symbol = "❌"
                print(f"{status_symbol} [{r.status}] {r.path}")
                if r.status == "DRIFT_DETECTED":
                    print(f"    Local Hash:    {r.local_hash}")
                    print(f"    Upstream Hash: {r.upstream_hash}")
                elif r.status == "ERROR":
                    print(f"    Error:         {r.error}")
                elif r.status == "UNTRACKED":
                    print(f"    No local file and no registry hash — run --sync to initialise.")

        if has_drift:
            return 1
        if has_error:
            return 2
        return 0

    if args.sync or args.dry_run:
        updated = 0
        wrote = 0
        for entry in entries:
            logger.info(f"Syncing {entry.path} from {entry.url}...")
            success, detail = sync_source(entry, _REPO_ROOT, dry_run=args.dry_run)
            if success:
                updated += 1
                if detail != NOT_MODIFIED_DETAIL:
                    wrote += 1

        if not args.dry_run and wrote > 0:
            save_registry(config_path, registry)
            logger.info("Regenerating manifest.json...")
            data_dir = _APP_ROOT / "data"
            generate_manifest(data_dir=data_dir)
            logger.info("Sync complete and manifest updated.")

        if entries and updated == 0:
            logger.error("All sources failed to sync.")
            return 2

        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
