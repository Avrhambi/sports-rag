"""Embed a query (Hebrew or otherwise) and retrieve the closest sport-tagged chunks.

Run as: python -m src.retrieve "<question>" [sport]
"""

import json
import re
from datetime import date
from functools import lru_cache

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import CHUNKS_PATH, EMBEDDING_MODEL_NAME, FAISS_INDEX_PATH

YEAR_PATTERN = re.compile(r"\b(?:19|20)\d{2}\b")
# A chunk whose year (or any year the query resolves to) is at stake gets
# guaranteed inclusion (see `retrieve`) rather than a soft score boost --
# aggregate questions ("how many times did Real Madrid win in the last 5
# years?") need every target year's full data (Match Info for the result,
# Lineups for the roster), not just whichever chunk happens to score highest.

# Hebrew number words for "the last N years" phrasing, e.g. "בשלוש השנים האחרונות".
_HEBREW_YEAR_COUNT_WORDS = {
    "שלוש": 3, "שלושה": 3,
    "ארבע": 4, "ארבעה": 4,
    "חמש": 5, "חמישה": 5,
}
_LAST_N_YEARS_DIGIT = re.compile(r"(\d+)\s*ה?שנים\s*ה?אחרונות")
_LAST_N_YEARS_WORD = re.compile(
    r"(" + "|".join(_HEBREW_YEAR_COUNT_WORDS) + r")\s*ה?שנים\s*ה?אחרונות"
)
_LAST_TWO_YEARS = re.compile(r"שנתיים\s*ה?אחרונות")
_THIS_YEAR = re.compile(r"\bהשנה\b")
_LAST_YEAR = re.compile(r"בשנה שעברה|שנה שעברה|אשתקד")
# No explicit count ("in recent years", "recently") -- default to a
# generous window since the whole seeded corpus is only 5 years deep.
_RECENT_YEARS_NO_COUNT = re.compile(r"בשנים\s*ה?אחרונות|לאחרונה")
_RECENT_YEARS_DEFAULT_N = 5


def most_recent_completed_final_year(today: date | None = None) -> int:
    """Both UCL and NBA finals are played in May/June. From July onward the
    next season is already underway and this calendar year's final has
    already happened; before July, last calendar year's final is the most
    recently completed one (a safe approximation around the fuzzy
    May/June boundary -- exact playoff calendars vary slightly by year)."""
    today = today or date.today()
    return today.year if today.month >= 7 else today.year - 1


def resolve_relative_years(query: str, anchor_year: int) -> set[str]:
    """Translate Hebrew relative-year phrases ("השנה", "שנה שעברה", "בשלוש
    השנים האחרונות") into absolute year strings anchored to `anchor_year`
    (the most recently completed final). Gemini isn't told what year it is
    unless the prompt says so, and this embedding model has no notion of
    "recent" at all, so relative time expressions need to be resolved to
    concrete years before they can inform retrieval."""
    years: set[str] = set()

    if _THIS_YEAR.search(query):
        years.add(str(anchor_year + 1))
    if _LAST_YEAR.search(query):
        years.add(str(anchor_year))
    if _LAST_TWO_YEARS.search(query):
        years.update(str(y) for y in range(anchor_year - 1, anchor_year + 1))

    word_match = _LAST_N_YEARS_WORD.search(query)
    if word_match:
        n = _HEBREW_YEAR_COUNT_WORDS[word_match.group(1)]
        years.update(str(y) for y in range(anchor_year - n + 1, anchor_year + 1))

    digit_match = _LAST_N_YEARS_DIGIT.search(query)
    if digit_match:
        n = int(digit_match.group(1))
        years.update(str(y) for y in range(anchor_year - n + 1, anchor_year + 1))

    if not years and _RECENT_YEARS_NO_COUNT.search(query):
        n = _RECENT_YEARS_DEFAULT_N
        years.update(str(y) for y in range(anchor_year - n + 1, anchor_year + 1))

    return years

# Likewise, competition-identifying keywords are a much more reliable sport
# signal than raw cosine similarity for this small, closed two-sport domain.
SPORT_KEYWORDS = {
    "football": ["כדורגל", "ליגת האלופות", "champions league", "uefa", "football", "soccer"],
    "basketball": ["כדורסל", "פיינלס", "nba", "basketball"],
}
SPORT_MATCH_BOOST = 0.15


def detect_sport(query: str) -> str | None:
    """Best-effort sport guess from competition-name keywords in the query."""
    lowered = query.lower()
    for sport_name, keywords in SPORT_KEYWORDS.items():
        if any(keyword in lowered for keyword in keywords):
            return sport_name
    return None


@lru_cache(maxsize=1)
def _load_model() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


@lru_cache(maxsize=1)
def _load_index_and_chunks() -> tuple[faiss.Index, list[dict]]:
    index = faiss.read_index(str(FAISS_INDEX_PATH))
    chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    return index, chunks


def retrieve(query: str, sport: str | None = None, top_k: int = 4) -> list[dict]:
    """Return the top_k chunks most similar to `query`, optionally filtered to one sport.

    The embedding model is multilingual, so a Hebrew query and the English chunk
    text land in the same vector space directly -- no translation step needed.
    """
    index, chunks = _load_index_and_chunks()
    model = _load_model()

    query_vec = np.asarray(model.encode([query], normalize_embeddings=True), dtype="float32")
    query_years = set(YEAR_PATTERN.findall(query))
    query_years |= resolve_relative_years(query, most_recent_completed_final_year())
    detected_sport = None if sport else detect_sport(query)
    effective_sport = sport or detected_sport
    needs_rerank = bool(sport or query_years or detected_sport)

    # Aggregate questions ("how many times did Real Madrid win in the last
    # 5 years?") need every target year's full match report -- Match Info
    # for the result, Lineups for the roster -- not just whichever chunks
    # happen to score highest. Guarantee all of them rather than relying on
    # a soft score boost that could still leave some years incomplete.
    guaranteed_ids = {
        chunk["id"]
        for chunk in chunks
        if query_years
        and any(year in chunk["source_title"] for year in query_years)
        and (not effective_sport or chunk["sport"] == effective_sport)
    }
    effective_top_k = max(top_k, len(guaranteed_ids))

    # Over-fetch the whole index whenever a sport filter or a score boost
    # needs to be applied, so top_k survives re-ranking rather than
    # truncating the raw FAISS order first.
    search_k = len(chunks) if needs_rerank else effective_top_k
    scores, indices = index.search(query_vec, search_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        chunk = chunks[idx]
        if sport and chunk["sport"] != sport:
            continue
        boosted_score = float(score)
        if chunk["id"] in guaranteed_ids:
            boosted_score += 1.0  # always sort above non-guaranteed chunks
        if detected_sport and chunk["sport"] == detected_sport:
            boosted_score += SPORT_MATCH_BOOST
        results.append({**chunk, "score": boosted_score})

    if needs_rerank:
        results.sort(key=lambda r: r["score"], reverse=True)
    return results[:effective_top_k]


if __name__ == "__main__":
    import sys

    query_arg = sys.argv[1] if len(sys.argv) > 1 else "מי ניצח בגמר ליגת האלופות 2022?"
    sport_arg = sys.argv[2] if len(sys.argv) > 2 else None
    for r in retrieve(query_arg, sport=sport_arg):
        print(f"[{r['sport']}] {r['score']:.3f}  {r['text'][:100]}")
