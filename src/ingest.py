"""Fetch UCL/NBA finals match reports and split them into tagged chunks.

Run as: python -m src.ingest
"""

import json
import re

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import (
    CHUNK_SIZE_WORDS,
    CHUNKS_PATH,
    EMBEDDING_MODEL_NAME,
    FACTS_PATH,
    FAISS_INDEX_PATH,
    RAW_DIR,
)
from src.ingest_basketball import build_basketball_docs
from src.ingest_football import build_football_docs


def chunk_markdown(markdown: str, max_words: int = CHUNK_SIZE_WORDS) -> list[str]:
    """Split a match-report Markdown doc into one chunk per `##` section, each
    prefixed with the document title for context. A section longer than
    `max_words` (e.g. a two-team lineup block) is further split on its `###`
    subheadings so a chunk never spans unrelated facts."""
    title_match = re.match(r"^# (.+)$", markdown, re.M)
    title = title_match.group(1) if title_match else ""

    chunks = []
    for section in re.split(r"^## ", markdown, flags=re.M)[1:]:
        heading, _, body = section.partition("\n")
        body = body.strip()
        if len(body.split()) <= max_words:
            chunks.append(f"{title} - {heading}\n\n{body}")
            continue

        subsections = re.split(r"^### ", body, flags=re.M)
        preamble = subsections[0].strip()
        if preamble:
            chunks.append(f"{title} - {heading}\n\n{preamble}")
        for sub in subsections[1:]:
            subheading, _, subbody = sub.partition("\n")
            chunks.append(f"{title} - {heading} - {subheading}\n\n{subbody.strip()}")

    return chunks


def build_chunks(docs: list[dict] | None = None) -> list[dict]:
    """Fetch every configured final and split its Markdown report into tagged chunks."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    docs = docs if docs is not None else build_football_docs() + build_basketball_docs()

    all_chunks = []
    for doc in docs:
        raw_path = RAW_DIR / f"{doc['source_title'].replace(' ', '_')}.md"
        raw_path.write_text(doc["markdown"], encoding="utf-8")

        for i, text in enumerate(chunk_markdown(doc["markdown"])):
            all_chunks.append(
                {
                    "id": f"{doc['sport']}-{doc['source_title']}-{i}",
                    "text": text,
                    "sport": doc["sport"],
                    "competition": doc["competition"],
                    "season": doc["season"],
                    "teams": doc["teams"],
                    "source_title": doc["source_title"],
                    "url": doc["url"],
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


def write_facts(docs: list[dict]) -> None:
    """Persist the structured per-final records alongside the prose chunks."""
    facts = [doc["facts"] for doc in docs if doc.get("facts")]
    FACTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    FACTS_PATH.write_text(json.dumps(facts, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    fetched_docs = build_football_docs() + build_basketball_docs()
    fetched_chunks = build_chunks(fetched_docs)
    print(f"Fetched and chunked {len(fetched_chunks)} chunks from {len(fetched_docs)} finals.")
    write_facts(fetched_docs)
    print(f"Wrote structured facts for {len(fetched_docs)} finals to {FACTS_PATH}.")
    build_index(fetched_chunks)
    print(f"Built FAISS index at {FAISS_INDEX_PATH} ({len(fetched_chunks)} vectors).")
