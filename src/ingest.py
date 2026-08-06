"""Fetch Wikipedia rule pages and split them into tagged chunks.

Run as: python -m src.ingest
"""

import json
import re

import faiss
import numpy as np
import requests
from sentence_transformers import SentenceTransformer

from src.config import (
    CHUNK_OVERLAP_WORDS,
    CHUNK_SIZE_WORDS,
    CHUNKS_PATH,
    EMBEDDING_MODEL_NAME,
    FAISS_INDEX_PATH,
    RAW_DIR,
    WIKIPEDIA_SOURCES,
)

WIKIPEDIA_API_URL = "https://en.wikipedia.org/w/api.php"
# Wikipedia's API rejects requests with no descriptive User-Agent (403).
REQUEST_HEADERS = {"User-Agent": "sports-rag/0.1 (https://github.com/Avrhambi/sports-rag)"}

STOP_HEADINGS = ("== References ==", "== See also ==", "== External links ==", "== Notes ==")


def fetch_wikipedia_plaintext(title: str) -> str:
    """Fetch the full plaintext extract of a Wikipedia article."""
    params = {
        "action": "query",
        "prop": "extracts",
        "explaintext": 1,
        "titles": title,
        "format": "json",
        "redirects": 1,
    }
    response = requests.get(WIKIPEDIA_API_URL, params=params, headers=REQUEST_HEADERS, timeout=30)
    response.raise_for_status()
    pages = response.json()["query"]["pages"]
    page = next(iter(pages.values()))
    if "extract" not in page:
        raise ValueError(f"No content found for Wikipedia page: {title!r}")
    return page["extract"]


def clean_text(text: str) -> str:
    """Strip trailing Wikipedia boilerplate sections and collapse extra whitespace."""
    for heading in STOP_HEADINGS:
        idx = text.find(heading)
        if idx != -1:
            text = text[:idx]
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE_WORDS, overlap: int = CHUNK_OVERLAP_WORDS) -> list[str]:
    """Split text into overlapping word-based chunks (approximates token windows)."""
    words = text.split()
    if not words:
        return []
    chunks = []
    step = max(chunk_size - overlap, 1)
    for start in range(0, len(words), step):
        chunk_words = words[start : start + chunk_size]
        if not chunk_words:
            break
        chunks.append(" ".join(chunk_words))
        if start + chunk_size >= len(words):
            break
    return chunks


def build_chunks() -> list[dict]:
    """Fetch every configured Wikipedia source and split it into sport-tagged chunks."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    all_chunks = []
    for source in WIKIPEDIA_SOURCES:
        title = source["title"]
        text = clean_text(fetch_wikipedia_plaintext(title))

        raw_path = RAW_DIR / f"{title.replace(' ', '_').replace('/', '_')}.txt"
        raw_path.write_text(text, encoding="utf-8")

        url = "https://en.wikipedia.org/wiki/" + title.replace(" ", "_")
        for i, chunk in enumerate(chunk_text(text)):
            all_chunks.append(
                {
                    "id": f"{source['sport']}-{i}",
                    "text": chunk,
                    "sport": source["sport"],
                    "source_title": title,
                    "url": url,
                }
            )
    return all_chunks


def build_index(chunks: list[dict]) -> None:
    """Embed chunk texts and persist a cosine-similarity FAISS index alongside their metadata."""
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    embeddings = model.encode([c["text"] for c in chunks], normalize_embeddings=True)
    embeddings = np.asarray(embeddings, dtype="float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    FAISS_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(FAISS_INDEX_PATH))
    CHUNKS_PATH.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    fetched_chunks = build_chunks()
    print(f"Fetched and chunked {len(fetched_chunks)} chunks from {len(WIKIPEDIA_SOURCES)} sources.")
    build_index(fetched_chunks)
    print(f"Built FAISS index at {FAISS_INDEX_PATH} ({len(fetched_chunks)} vectors).")
