# Sports Rules RAG

Cross-lingual RAG assistant: users ask sports-rules questions **in Hebrew**, the
system retrieves grounding passages from **English** Wikipedia rule pages
(football, basketball) and returns a grounded Hebrew answer with sources.

## Key files

- `src/config.py` — model names, Wikipedia source list, file paths, env loading
- `src/ingest.py` — fetch Wikipedia pages → chunk → embed → build FAISS index (offline, rerunnable)
- `src/retrieve.py` — embed a Hebrew query, FAISS search, sport filtering
- `src/generate.py` — Gemini prompt that answers in Hebrew grounded only in retrieved chunks
- `app.py` — FastAPI app: `/api/ask`, `/api/health`, serves `static/`
- `static/` — hand-written pitch/court-themed UI (no template framework)
- `eval/` — seed Hebrew Q&A set + scoring script

## Run commands

```
pip install -r requirements.txt
cp .env.example .env   # fill in GEMINI_API_KEY
python -m src.ingest   # builds data/faiss.index + data/chunks.json
uvicorn app:app --reload
python eval/eval.py    # scores the pipeline against eval/qa_testset.json
```

## Architecture notes

- Embedding model is `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
  (local, multilingual, 384-dim) — the same model embeds both the Hebrew query
  and the English source chunks into one shared space, so there is no
  translation step.
- FAISS index is `IndexFlatIP` over L2-normalized vectors (cosine similarity);
  exact search is fine at this corpus size.
- `data/faiss.index` and `data/chunks.json` are generated, gitignored, and
  rebuilt via `python -m src.ingest` — never hand-edited.
- Generation is grounded: the Gemini prompt in `src/generate.py` is instructed
  to answer only from retrieved chunks and say so if they don't cover the
  question, to avoid hallucinated rules.
- UI intentionally avoids chat-app/card conventions — it's laid out like a
  playing surface (halfway line, center circle, penalty box) per the
  pitch-lines design direction; see `static/styles.css`.
