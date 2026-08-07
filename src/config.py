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

EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
GEMINI_MODEL_NAME = "gemini-3.1-flash-lite"
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

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
