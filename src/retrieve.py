"""Embed a query (Hebrew or otherwise) and retrieve the closest sport-tagged chunks.

Run as: python -m src.retrieve "<question>" [sport]
"""

import json
from functools import lru_cache

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import CHUNKS_PATH, EMBEDDING_MODEL_NAME, FAISS_INDEX_PATH


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

    # Over-fetch the whole index when filtering by sport, so top_k survives the filter.
    search_k = len(chunks) if sport else top_k
    scores, indices = index.search(query_vec, search_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        chunk = chunks[idx]
        if sport and chunk["sport"] != sport:
            continue
        results.append({**chunk, "score": float(score)})
        if len(results) >= top_k:
            break
    return results


if __name__ == "__main__":
    import sys

    query_arg = sys.argv[1] if len(sys.argv) > 1 else "כמה שחקנים יש בקבוצת כדורגל?"
    sport_arg = sys.argv[2] if len(sys.argv) > 2 else None
    for r in retrieve(query_arg, sport=sport_arg):
        print(f"[{r['sport']}] {r['score']:.3f}  {r['text'][:100]}")
