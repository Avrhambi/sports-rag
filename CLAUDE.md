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
  report into section-aware chunks, embeds, builds the FAISS index, and writes
  `data/facts.json` (offline, rerunnable)
- `src/facts.py` — deterministic aggregates (titles, margins, attendance
  ranks, per-player point totals) over `data/facts.json`
- `src/plan.py` — turns a question into a `QueryPlan` (sport, years, intent)
  via Gemini, with a frozen regex/keyword fallback for offline use
- `src/retrieve.py` — embed a Hebrew query, FAISS search, sport filtering, plus
  plan-driven guaranteed inclusion and a re-rank boost (see Architecture notes)
- `src/generate.py` — Gemini prompt that answers in Hebrew grounded only in
  retrieved chunks, resolving every question into one of three states:
  supported, false-by-closure (absent from a complete set is an answer, not a
  gap), or outside coverage
- `app.py` — FastAPI app: `/api/ask`, `/api/health`, serves `static/`;
  `resolve_sport()` decides between the UI's sport tab and the question
- `static/` — hand-written pitch/court-themed UI (no template framework)
- `eval/` — seed Hebrew Q&A set (one fact per seeded final) + scoring script

## Run commands

```
pip install -r requirements.txt
cp .env.example .env   # fill in GEMINI_API_KEY_1 (a 2nd key doubles daily quota)
python -m src.ingest   # builds data/faiss.index + data/chunks.json
uvicorn app:app --reload
python eval/eval.py    # scores the pipeline against eval/qa_testset.json
python -m eval.eval --as-typed   # the same questions, phrased the way users type them
```

## Architecture notes

- **The Gemini model is part of the system, not a swappable backend.** Both
  the planner and the answerer are Gemini calls, so changing
  `GEMINI_MODEL_NAME` changes behaviour and retires the eval numbers. This
  system targets `gemini-3.1-flash-lite`, which scores 1.00 correctness on
  all 33 as-typed questions. `gemini-3.5-flash-lite` was tried and rejected:
  0.94, with two regressions 3.1 does not have — its planner stopped
  expanding a bare "מי ניצח בגמר" to every covered year (dropping retrieval
  to an arbitrary top-4), and it reported Boston as losing the 2024 NBA
  Finals, which Boston won, in five runs out of six. `eval.eval` prints the
  model at the top of every run so no result is ever compared across models
  by accident. Work parked from that experiment lives on the branch
  `robustness/model-agnostic-joins`, unvalidated on 3.1 and unmerged.

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
- Query planning in `src/plan.py`: retrieval needs to know which sport, which
  years, and whether one chunk can answer the question at all — none of which
  the embedding gives you. One Gemini call returns all three as a `QueryPlan`.
  It replaced a growing pile of hand-written Hebrew regexes for relative-year
  phrases; those still exist as `heuristic_plan`, the fallback used when no
  API key is set (so `eval.py`'s retrieval tier still runs offline) or when
  the planner call fails, but they are frozen — new phrasings belong in the
  planner prompt, not in another regex. The fallback always reports `factoid`
  intent, so callers degrade to plain top-k rather than guessing.
- **Questions arrive under-specified, and there is no default sport.** Real
  questions leave things out that the eval set never did: no question mark,
  no geresh in transliterated names ("גיילן" for "ג'יילן"), "ב5" for "ב-5",
  and above all no competition — "מי אימן את האלופה ב2024", "מי ניצח בגמר".
  Planning and retrieval turned out to be robust to every one of those except
  the missing competition, where the planner picked a sport on a hunch: it
  chose football for the coach question, a sport this corpus holds no coach
  aggregates for, so coverage was 0% and the user got a refusal to a question
  both sports answer. The planner prompt now says outright that there is no
  default, and `src/generate.py` answers for both competitions and labels
  each. Keep the disambiguators that genuinely work (a UCL final is one match,
  an NBA Finals is a series) — the goal is not to widen everything to "any".
  `eval/qa_testset.json` carries an `as_typed` form of every question and
  `python -m eval.eval --as-typed` re-runs the set in it, against the same
  keywords and reference answers, so the two runs compare directly.
- The UI's sport tab is a scope, and an off-tab question is answered as such.
  `off_tab_sport()` in `app.py` detects that the plan's sport is not the
  selected tab and returns, before retrieval, the one answer Python writes
  itself: what this tab covers, what the question was about, and how to get
  an answer. Searching past the mismatch instead — pushing a UCL question
  through the basketball filter — produced a refusal phrased as missing data,
  which blamed the corpus for what the tab caused. It also spends one Gemini
  call instead of two. `off_tab_sport` in the response drives a one-click
  switch-and-re-ask in the UI.
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
- **Pre-join in Python; never make the model derive a relationship.** Every
  wrong fact this system has produced came from asking it to derive rather
  than read: it reversed scorelines re-narrating `0–1` winner-first, said a
  player never won while holding his roster and his team's title in separate
  chunks, and put a team among one year's losers because the tally carried
  counts but no years. So `build_facts()` emits a winner-first scoreline,
  `team_result` on every person, and tallies with their years, and
  `src/facts.py` renders rows shaped like the answer rather than columns to
  be joined. Two things this cost us to learn: a computed line must be
  unmistakable for an adjacent one (a multi-year points sum printed beside a
  single-game high got returned as the answer to "most points in a game"),
  and **the join has to run in the direction the question asks**. A per-year
  squad list was verified present in the prompt and the model still answered
  "Tatum never won" — it scanned for the most answer-shaped row, found the
  MVP column, and concluded from that. `_individual_honours` therefore emits
  one row per person, not one row per year.
- Two parallel tracks come out of ingestion: prose chunks for retrieval, and
  `data/facts.json` — one flat structured record per final (result, margin,
  attendance, per-player points) built by `build_facts()` in each ingester.
  `src/facts.py` turns those into a block of pre-computed totals and rankings
  that `src/generate.py` prepends to the context for non-factoid intents, so
  a count or a superlative never depends on the model adding up a box score.
  Deliberately not a query engine: at ten finals every aggregate worth asking
  fits in one block, so there is no SQL and nothing to plan at answer time.
- `data/faiss.index`, `data/chunks.json` and `data/facts.json` are generated,
  gitignored, and rebuilt via `python -m src.ingest` — never hand-edited.
- Generation is grounded: the Gemini prompt in `src/generate.py` is instructed
  to answer only from retrieved chunks and say so if they don't cover the
  question, to avoid hallucinated match facts.
