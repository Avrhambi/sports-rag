# Sports Finals History RAG

Ask questions **in Hebrew** about historical **UEFA Champions League** and
**NBA Finals** matches — winners, venues, referees/officials, lineups, key
events, box scores — and get answers grounded in structured match-report
data, with sources you can expand to check the original text yourself.
Seeded with the last 5 finals of each competition (2022–2026).

The UI is laid out as a top-down pitch/court diagram rather than a generic
chat window: the question box doubles as the center circle, answers render
inside a penalty-box-style frame, and sources appear as substitution-board
tiles. Selecting a sport swaps the whole page between turf green and
hardwood amber.

## How it works

```
Wikipedia wikitext (UCL finals) --\                 /--> Markdown match reports --chunk (by section)--> embed (MiniLM) --> FAISS index
                                    +--ingest------+
nba_api + Wikipedia infobox (NBA) -/                \--> structured per-final records ------------------------------> data/facts.json

Hebrew question --> plan (Gemini): which sport, which years, factoid or aggregate?
                          |
                          +--> FAISS search, filtered by the plan; every planned year's chunks are guaranteed in
                          |
                          +--> for non-factoid intents only: pre-computed totals and rankings from facts.json
                                       |
                                       v
                    Gemini: answer in Hebrew, grounded only in the retrieved EN chunks
                                       |
                    Hebrew answer + sources --> UI
```

Counting, ranking and summing are done in Python over `data/facts.json`, not
by the model — the model gets the totals as another source and writes the
answer around them.

Embedding model: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
(local, multilingual, no API key) — the same model embeds the Hebrew query
and the English chunks into one shared space, so there's no translation step.

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate        # .venv\Scripts\Activate.ps1 on Windows PowerShell
pip install -r requirements.txt
cp .env.example .env          # fill in GEMINI_API_KEY_1
```

## Run

```bash
python -m src.ingest          # fetch + chunk + embed -> data/faiss.index, data/chunks.json, data/facts.json
uvicorn app:app --reload      # http://127.0.0.1:8000
```

## Test & evaluate

```bash
pytest                        # unit tests (Markdown chunking)
python -m eval.eval           # retrieval + generation quality scores against eval/qa_testset.json
python -m eval.eval --as-typed  # the same questions phrased the way users type them
```

`eval.eval` always runs retrieval checks (sport-match accuracy, expected-keyword
coverage — no API key needed). With a key set, it also asks Gemini
to judge each generated answer's faithfulness and relevance.

`--as-typed` re-asks every question in its `as_typed` form — no question mark,
no geresh in names, "ב5" for "ב-5", often no competition named — against the
same expected keywords and reference answers, so the two runs compare directly
and the difference is what the phrasing cost.

## Key files

See [CLAUDE.md](CLAUDE.md) for the full file-by-file breakdown and architecture notes.
