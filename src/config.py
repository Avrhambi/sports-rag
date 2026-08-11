"""Shared configuration for ingestion, retrieval, and generation."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
FAISS_INDEX_PATH = DATA_DIR / "faiss.index"
CHUNKS_PATH = DATA_DIR / "chunks.json"
# Structured per-final records, parallel to the prose chunks: what counting
# and ranking read instead of re-deriving numbers from the Markdown reports.
FACTS_PATH = DATA_DIR / "facts.json"

EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
GEMINI_MODEL_NAME = "gemini-3.1-flash-lite"

def _gemini_keys() -> list[str]:
    """Every configured key, in the order they should be tried.

    Free-tier quota is per key per day, so a second key doubles how much
    evaluation fits in a day -- a full eval run is ~3 calls per question and
    the limit is 500. `GEMINI_API_KEY` (unnumbered) is still honoured so an
    existing .env keeps working.
    """
    numbered = [os.environ.get(f"GEMINI_API_KEY_{i}", "") for i in range(1, 5)]
    keys = [*numbered, os.environ.get("GEMINI_API_KEY", "")]
    seen: list[str] = []
    for key in (k.strip() for k in keys):
        if key and key not in seen:
            seen.append(key)
    return seen


GEMINI_API_KEYS = _gemini_keys()
# Kept for "is Gemini configured at all" checks.
GEMINI_API_KEY = GEMINI_API_KEYS[0] if GEMINI_API_KEYS else ""

# Word-based fallback chunk size for any prose section too long to keep as one chunk.
CHUNK_SIZE_WORDS = 400
CHUNK_OVERLAP_WORDS = 60

# Add more years here to extend coverage; each maps to a
# "<year> UEFA Champions League final" Wikipedia page.
UCL_FINALS_YEARS = [2022, 2023, 2024, 2025, 2026]

# Add more years here to extend coverage; each maps to a
# "<year> NBA Finals" series (clinching game is fetched via nba_api).
NBA_FINALS_YEARS = [2022, 2023, 2024, 2025, 2026]

SPORTS = ["football", "basketball"]

# What the corpus contains, stated explicitly so the answerer can tell "absent
# from a complete set" (an answer) from "outside what we hold" (a refusal).
# `unit` is the honest part: an NBA Finals report is one game of a series, so
# anything per-series is outside coverage no matter how good retrieval is.
CORPUS_COVERAGE = {
    "football": {
        "competition": "UEFA Champions League final",
        "years": UCL_FINALS_YEARS,
        "unit": "the final match itself, with both full squads",
    },
    "basketball": {
        "competition": "NBA Finals",
        "years": NBA_FINALS_YEARS,
        "unit": "the series-clinching game only, with both full box scores",
    },
}


def coverage_summary(sport: str | None = None) -> str:
    """One line per covered competition, for the answerer's prompt."""
    lines = []
    for name, info in CORPUS_COVERAGE.items():
        if sport and name != sport:
            continue
        years = ", ".join(str(y) for y in info["years"])
        lines.append(f"- {info['competition']}: {years} — {info['unit']}")
    return "\n".join(lines)
