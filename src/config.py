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
GEMINI_MODEL_NAME = "gemini-2.5-flash"
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

CHUNK_SIZE_WORDS = 400
CHUNK_OVERLAP_WORDS = 60

# Add more (title, sport) pairs here to extend coverage.
WIKIPEDIA_SOURCES = [
    {"title": "Laws of the Game (association football)", "sport": "football"},
    {"title": "Rules of basketball", "sport": "basketball"},
]

SPORTS = ["football", "basketball"]
