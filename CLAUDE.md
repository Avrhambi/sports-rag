# Sports Finals History RAG

Cross-lingual RAG assistant: users ask **Hebrew** questions about historical
**UEFA Champions League** and **NBA Finals** matches — winners, venues,
referees/officials, lineups, key events, box scores — and the system retrieves
grounding passages from structured match-report data and returns a grounded
Hebrew answer with sources. Currently seeded with the last 5 finals of each
competition (2022–2026); extend by adding years to `UCL_FINALS_YEARS` /
`NBA_FINALS_YEARS` in `src/config.py`.

## Key files

- `src/config.py` — model names, seeded finals years, file paths, env loading
- `src/wikitext.py` — shared MediaWiki fetch + generic wikitext-cleanup helpers
- `src/ingest_football.py` — parses each UCL final's Wikipedia wikitext (infobox,
  lineup table, statistics) into a structured Markdown match report
- `src/ingest_basketball.py` — pulls each NBA Finals clinching game's box score
  from `nba_api`, plus head coaches/MVP from a small Wikipedia infobox lookup
  (not exposed by any nba_api endpoint)
- `src/ingest.py` — orchestrator: calls both ingesters, splits each Markdown
  report into section-aware chunks, embeds, builds the FAISS index (offline,
  rerunnable)
- `src/retrieve.py` — embed a Hebrew query, FAISS search, sport filtering, plus
  a year/competition-keyword re-rank boost (see Architecture notes)
- `src/generate.py` — Gemini prompt that answers in Hebrew grounded only in
  retrieved chunks
- `app.py` — FastAPI app: `/api/ask`, `/api/health`, serves `static/`
- `static/` — hand-written pitch/court-themed UI (no template framework)
- `eval/` — seed Hebrew Q&A set (one fact per seeded final) + scoring script

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
- No scraping library is used for football data: `soccerdata` was tried and
  dropped — none of its scrapers (FBref, ESPN, WhoScored, Sofascore) cover
  UEFA Champions League, and its FBref path requires a full Chrome browser via
  seleniumbase. UCL finals are parsed directly from Wikipedia wikitext instead
  (`src/wikitext.py` + `src/ingest_football.py`), reusing the project's
  existing `requests` dependency with no browser involved.
- Each match report is one Markdown doc with `##` sections (Match Info,
  Lineups, Key Events/Box Score, Statistics). `chunk_markdown()` in
  `src/ingest.py` splits on those section headers — not a word-sliding
  window — so a chunk never mixes unrelated facts; a section that runs long
  (e.g. a two-team lineup block) falls back to splitting on `###`
  subheadings.
- Retrieval re-ranking in `src/retrieve.py`: this embedding model doesn't
  reliably discriminate a specific year, or football vs. basketball, across
  near-identical templated finals reports (5 UCL finals / 5 NBA Finals that
  differ mainly by year and team names). An explicit year or competition
  keyword (e.g. "ליגת האלופות", "NBA") in the query gets a small score boost
  toward matching chunks — found and fixed via `eval.py`'s unfiltered
  top-sport-match metric, which exists specifically to catch this failure
  mode. Also avoid very short, mostly-numeric standalone chunks (e.g. a bare
  quarter-score table) — they can score anomalously high against unrelated
  queries; fold that data into a richer text chunk instead.
- `data/faiss.index` and `data/chunks.json` are generated, gitignored, and
  rebuilt via `python -m src.ingest` — never hand-edited.
- Generation is grounded: the Gemini prompt in `src/generate.py` is instructed
  to answer only from retrieved chunks and say so if they don't cover the
  question, to avoid hallucinated match facts.
