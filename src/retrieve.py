"""Embed a query (Hebrew or otherwise) and retrieve the closest sport-tagged chunks.

Run as: python -m src.retrieve "<question>" [sport]
"""

import json
import re
from functools import lru_cache

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import CHUNKS_PATH, EMBEDDING_MODEL_NAME, FAISS_INDEX_PATH

YEAR_PATTERN = re.compile(r"\b(?:19|20)\d{2}\b")
# The embedding model doesn't reliably discriminate a specific year across
# near-identical templated match reports (5 UCL finals, 5 NBA Finals that
# differ mainly by year/teams), so an explicit year mention in the query is
# boosted toward chunks from that year rather than left to raw cosine score.
YEAR_MATCH_BOOST = 0.15

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
    detected_sport = None if sport else detect_sport(query)
    needs_rerank = bool(sport or query_years or detected_sport)

    # Over-fetch the whole index whenever a sport filter or a score boost
    # needs to be applied, so top_k survives re-ranking rather than
    # truncating the raw FAISS order first.
    search_k = len(chunks) if needs_rerank else top_k
    scores, indices = index.search(query_vec, search_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        chunk = chunks[idx]
        if sport and chunk["sport"] != sport:
            continue
        boosted_score = float(score)
        if query_years and any(year in chunk["source_title"] for year in query_years):
            boosted_score += YEAR_MATCH_BOOST
        if detected_sport and chunk["sport"] == detected_sport:
            boosted_score += SPORT_MATCH_BOOST
        results.append({**chunk, "score": boosted_score})

    if needs_rerank:
        results.sort(key=lambda r: r["score"], reverse=True)
    return results[:top_k]


if __name__ == "__main__":
    import sys

    query_arg = sys.argv[1] if len(sys.argv) > 1 else "מי ניצח בגמר ליגת האלופות 2022?"
    sport_arg = sys.argv[2] if len(sys.argv) > 2 else None
    for r in retrieve(query_arg, sport=sport_arg):
        print(f"[{r['sport']}] {r['score']:.3f}  {r['text'][:100]}")
