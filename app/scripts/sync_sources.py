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


@dataclass
class SourceEntry:
    path: str
    url: str
    type: str
    selector: str | None = None
    category: str = "general"
    content_hash: str | None = None
    last_synced: str | None = None


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


class HTMLContentExtractor(HTMLParser):
    """
    Robust HTML to Markdown content extractor.
    Strips scripts, styles, layout navigation, and extracts targeted containers.
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
        self.inside_target = target_selector is None
        self.selector_found = target_selector is None
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

        # Check selector entry condition
        if not self.inside_target and self.target_selector:
            if self._matches_selector(tag_lower, attr_dict):
                self.inside_target = True
                self.selector_found = True
                self.selector_depth = 1
                return

        if self.inside_target and self.target_selector and self.selector_depth > 0:
            if tag_lower not in self.VOID_TAGS:
                self.selector_depth += 1

        if not self.inside_target:
            return

        # Heading tags
        if re.match(r"^h[1-6]$", tag_lower):
            self.heading_level = int(tag_lower[1])
            self.tokens.append(f"\n\n{'#' * self.heading_level} ")
        elif tag_lower in self.BLOCK_TAGS:
            self.tokens.append("\n\n")
        elif tag_lower == "br":
            self.tokens.append("\n")
        elif tag_lower == "li":
            self.tokens.append("\n- ")
        elif tag_lower in ("strong", "b"):
            self.is_bold = True
            self.tokens.append("**")
        elif tag_lower in ("em", "i"):
            self.is_italic = True
            self.tokens.append("*")
        elif tag_lower == "a":
            href = attr_dict.get("href", "")
            if href and not href.startswith("javascript:"):
                self.current_link = href

        if tag_lower not in self.VOID_TAGS:
            self.tag_stack.append(tag_lower)


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
                self.inside_target = False
                return


        if not self.inside_target:
            return

        if re.match(r"^h[1-6]$", tag_lower):
            self.heading_level = 0
            self.tokens.append("\n\n")
        elif tag_lower in self.BLOCK_TAGS:
            self.tokens.append("\n\n")
        elif tag_lower in ("strong", "b"):
            self.is_bold = False
            self.tokens.append("**")
        elif tag_lower in ("em", "i"):
            self.is_italic = False
            self.tokens.append("*")
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

        text = data
        if not text:
            return

        if self.current_link:
            self.link_text_tokens.append(text)
            return

        self.tokens.append(text)

    def _matches_selector(self, tag: str, attrs: dict[str, str]) -> bool:
        if not self.target_selector:
            return False
        selector = self.target_selector.strip()
        if selector.startswith("#"):
            target_id = selector[1:]
            return attrs.get("id", "") == target_id
        if selector.startswith("."):
            target_class = selector[1:]
            classes = attrs.get("class", "").split()
            return target_class in classes
        return tag == selector.lower()

    def get_markdown(self) -> str:
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
        raise SelectorNotFoundError(extractor.target_selector)


def clean_bclaws_content(raw_html: str, selector: str | None) -> str:
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
    content = re.sub(r"\n{3,}", "\n\n", content)
    return content.strip()


def extract_content(raw_html: str, doc_type: str, selector: str | None) -> tuple[str, str]:
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
        body = clean_bclaws_content(raw_html, selector=resolved_selector)
    else:
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


def compute_content_hash(text: str) -> str:
    """Compute SHA-256 hash of normalized text."""
    normalized = re.sub(r"\s+", " ", text).strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def format_provenance_header(
    title: str,
    url: str,
    upstream_last_modified: str | None = None,
    ingestion_date: str | None = None,
) -> str:
    """Formats canonical provenance metadata header."""
    last_mod = upstream_last_modified or "Unknown"
    ingest = ingestion_date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return (
        f"# {title}\n\n"
        f"**Source:** [{title}]({url})  \n"
        f"**Upstream Last Modified:** {last_mod}  \n"
        f"**Ingestion Date:** {ingest}  \n\n"
    )


def fetch_upstream(url: str, timeout: int = 15) -> tuple[str, dict[str, str]]:
    """Fetches upstream content and response headers."""
    req = urllib.request.Request(
        url,
        headers={"User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        content = resp.read().decode("utf-8", errors="replace")
        headers = {k.lower(): v for k, v in resp.headers.items()}
        return content, headers


def load_registry(config_path: Path) -> list[SourceEntry]:
    """Loads declarative source registry from sources.yaml."""
    if not config_path.exists():
        raise FileNotFoundError(f"Source registry not found: {config_path}")
    raw_data = SimpleYamlLoader.load(config_path.read_text(encoding="utf-8"))
    entries: list[SourceEntry] = []
    for s in raw_data.get("sources", []):
        entries.append(
            SourceEntry(
                path=s["path"],
                url=s["url"],
                type=s.get("type", "html_selector"),
                selector=s.get("selector"),
                category=s.get("category", "general"),
                content_hash=s.get("content_hash"),
                last_synced=s.get("last_synced"),
            )
        )
    return entries


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
    header = leading_comment_block(existing)
    data = {
        "sources": [asdict(e) for e in entries]
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
    local_file = repo_root / source.path
    local_substantive = ""
    local_hash = source.content_hash

    if local_file.exists():
        local_text = local_file.read_text(encoding="utf-8")
        local_substantive = extract_substantive_body(local_text)
        local_hash = compute_content_hash(local_substantive)
    elif source.content_hash is None:
        return DriftResult(
            path=source.path,
            url=source.url,
            status="UNTRACKED",
        )

    try:
        raw_html, headers = fetch_upstream(source.url)
        title, upstream_body = extract_content(raw_html, source.type, source.selector)
        upstream_substantive = extract_substantive_body(upstream_body)
        if not upstream_substantive:
            return DriftResult(
                path=source.path,
                url=source.url,
                status="ERROR",
                local_hash=local_hash,
                error="Upstream extraction produced no substantive content",
            )
        upstream_hash = compute_content_hash(upstream_substantive)
        upstream_last_modified = headers.get("last-modified")

        if source.content_hash and upstream_hash != source.content_hash:
            return DriftResult(
                path=source.path,
                url=source.url,
                status="DRIFT_DETECTED",
                local_hash=local_hash or source.content_hash,
                upstream_hash=upstream_hash,
                upstream_last_modified=upstream_last_modified,
            )
        elif local_hash and upstream_hash != local_hash:
            return DriftResult(
                path=source.path,
                url=source.url,
                status="DRIFT_DETECTED",
                local_hash=local_hash,
                upstream_hash=upstream_hash,
                upstream_last_modified=upstream_last_modified,
            )

        return DriftResult(
            path=source.path,
            url=source.url,
            status="MATCH",
            local_hash=local_hash,
            upstream_hash=upstream_hash,
            upstream_last_modified=upstream_last_modified,
        )

    except (urllib.error.URLError, TimeoutError) as e:
        logger.warning(f"Network error checking {source.url}: {e}")
        return DriftResult(
            path=source.path,
            url=source.url,
            status="ERROR",
            local_hash=local_hash,
            error=f"Network error: {e}",
        )
    except Exception as e:
        logger.error(f"Failed to check {source.url}: {e}", exc_info=True)
        return DriftResult(
            path=source.path,
            url=source.url,
            status="ERROR",
            local_hash=local_hash,
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
    try:
        raw_html, headers = fetch_upstream(source.url)
        title, body = extract_content(raw_html, source.type, source.selector)
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

    entries = load_registry(config_path)
    if args.filter:
        entries = [e for e in entries if args.filter in e.path or args.filter in e.url]
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
        for entry in entries:
            logger.info(f"Syncing {entry.path} from {entry.url}...")
            success, _ = sync_source(entry, _REPO_ROOT, dry_run=args.dry_run)
            if success:
                updated += 1

        if not args.dry_run and updated > 0:
            save_registry(config_path, entries)
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
