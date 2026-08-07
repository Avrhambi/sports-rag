"""Shared helpers for pulling and cleaning MediaWiki wikitext.

Used by both ingest_football.py (full match-report parsing) and
ingest_basketball.py (a small infobox lookup for data nba_api doesn't
expose, like head coaches). Sport-specific templates (e.g. football's
{{goal|..}}) are handled by the caller before delegating to clean_wikitext
for the generic markup cleanup.
"""

import re

import requests

WIKIPEDIA_API_URL = "https://en.wikipedia.org/w/api.php"
REQUEST_HEADERS = {"User-Agent": "sports-rag/0.1 (https://github.com/Avrhambi/sports-rag)"}


def fetch_wikitext(title: str) -> str:
    """Fetch the raw wikitext of the current revision of a Wikipedia article."""
    params = {
        "action": "query",
        "prop": "revisions",
        "rvprop": "content",
        "rvslots": "main",
        "titles": title,
        "format": "json",
        "redirects": 1,
    }
    response = requests.get(WIKIPEDIA_API_URL, params=params, headers=REQUEST_HEADERS, timeout=30)
    response.raise_for_status()
    pages = response.json()["query"]["pages"]
    page = next(iter(pages.values()))
    if "missing" in page:
        raise ValueError(f"No Wikipedia page found for {title!r}")
    return page["revisions"][0]["slots"]["main"]["*"]


def extract_section(wikitext: str, heading: str) -> str:
    """Return the text of a `==Heading==`/`===Heading===` section, up to the next
    heading of the same or shallower level."""
    match = re.search(rf"^(=+)\s*{re.escape(heading)}\s*\1\s*$", wikitext, re.M)
    if not match:
        raise ValueError(f"Section {heading!r} not found")
    level = len(match.group(1))
    start = match.end()
    next_heading = re.search(rf"^={{2,{level}}}[^=].*$", wikitext[start:], re.M)
    end = start + next_heading.start() if next_heading else len(wikitext)
    return wikitext[start:end]


def clean_wikitext(text: str) -> str:
    """Strip generic wikitext markup (refs, links, bold/italic, leftover
    templates) down to plain, readable text."""
    text = re.sub(r"<ref[^>]*>.*?</ref>", "", text, flags=re.S)
    text = re.sub(r"<ref[^>]*/>", "", text)
    # Fallback: drop any remaining templates (a few levels of nesting).
    for _ in range(3):
        text = re.sub(r"\{\{[^{}]*\}\}", "", text)
    text = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]", r"\1", text)
    text = re.sub(r"'''(.*?)'''", r"\1", text)
    text = re.sub(r"''(.*?)''", r"\1", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
