"""Embed a query (Hebrew or otherwise) and retrieve the closest sport-tagged chunks.

What the query *means* -- which sport, which years, whether one chunk can even
answer it -- is worked out in `src.plan`; this module only searches.

Run as: python -m src.retrieve "<question>" [sport]
"""

import json
from functools import lru_cache

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import CHUNKS_PATH, EMBEDDING_MODEL_NAME, FAISS_INDEX_PATH
from src.plan import QueryPlan, plan_query

# A chunk whose year the plan asks for gets guaranteed inclusion (see
# `retrieve`) rather than a soft score boost -- aggregate questions ("how many
# times did Real Madrid win in the last 5 years?") need every target year's
# full data (Match Info for the result, Lineups for the roster), not just
# whichever chunk happens to score highest.

# Likewise, the plan's sport is a much more reliable signal than raw cosine
# similarity for this small, closed two-sport domain.
SPORT_MATCH_BOOST = 0.15


@lru_cache(maxsize=1)
def _load_model() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


@lru_cache(maxsize=1)
def _load_index_and_chunks() -> tuple[faiss.Index, list[dict]]:
    index = faiss.read_index(str(FAISS_INDEX_PATH))
    chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    return index, chunks


def retrieve(
    query: str,
    sport: str | None = None,
    top_k: int = 4,
    plan: QueryPlan | None = None,
) -> list[dict]:
    """Return the top_k chunks most similar to `query`, optionally filtered to one sport.

    The embedding model is multilingual, so a Hebrew query and the English chunk
    text land in the same vector space directly -- no translation step needed.

    An explicit `sport` (from the UI's sport tabs) overrides the plan's guess.
    Pass `plan` to reuse a plan the caller already has rather than paying for a
    second planner call.
    """
    index, chunks = _load_index_and_chunks()
    model = _load_model()
    plan = plan or plan_query(query)

    query_vec = np.asarray(model.encode([query], normalize_embeddings=True), dtype="float32")
    query_years = plan.year_strings
    detected_sport = None if sport else plan.sport_filter
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
    query_plan = plan_query(query_arg)
    print(f"plan: {query_plan}")
    for r in retrieve(query_arg, sport=sport_arg, plan=query_plan):
        print(f"[{r['sport']}] {r['score']:.3f}  {r['text'][:100]}")
