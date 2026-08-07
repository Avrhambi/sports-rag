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
Wikipedia wikitext (UCL finals) --\
                                    +--ingest--> Markdown match reports --chunk (by section)--> embed (MiniLM) --> FAISS index
nba_api + Wikipedia infobox (NBA) -/                                                                                    |
                                                                                                                          |
Hebrew question --embed (same MiniLM)--> FAISS search (+ sport filter, + year/competition re-rank boost)
                                                                                                                          |
                                                                    Gemini: answer in Hebrew, grounded
                                                                    only in the retrieved EN chunks
                                                                                                                          |
                                                                    Hebrew answer + sources --> UI
```

Embedding model: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
(local, multilingual, no API key) — the same model embeds the Hebrew query
and the English chunks into one shared space, so there's no translation step.

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate        # .venv\Scripts\Activate.ps1 on Windows PowerShell
pip install -r requirements.txt
cp .env.example .env          # fill in GEMINI_API_KEY
```

## Run

```bash
python -m src.ingest          # fetch + chunk + embed -> data/faiss.index, data/chunks.json
uvicorn app:app --reload      # http://127.0.0.1:8000
```

## Test & evaluate

```bash
pytest                        # unit tests (Markdown chunking)
python -m eval.eval           # retrieval + generation quality scores against eval/qa_testset.json
```

`eval.eval` always runs retrieval checks (sport-match accuracy, expected-keyword
coverage — no API key needed). With `GEMINI_API_KEY` set, it also asks Gemini
to judge each generated answer's faithfulness and relevance.

## Key files

See [CLAUDE.md](CLAUDE.md) for the full file-by-file breakdown and architecture notes.
