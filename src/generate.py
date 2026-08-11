"""Generate a grounded Hebrew answer from retrieved English chunks via Gemini."""

from datetime import date

from google import genai

from src.config import GEMINI_API_KEY, GEMINI_MODEL_NAME, coverage_summary
from src.facts import derived_facts_block
from src.plan import QueryPlan

# The prompt's central job is to give the model three states instead of two.
# The old version offered only "answer" or "say you can't", so a complete
# winners table that simply doesn't mention Barcelona read as missing data,
# and every closed-world question collapsed into a refusal.
SYSTEM_PROMPT = """You are a sports history assistant. Answer strictly in Hebrew.

WHAT THE CORPUS HOLDS
{coverage}
Within that scope the corpus is COMPLETE: every listed final is present, with
its result, its participating teams and both full squads. Nothing outside that
scope exists in it.

HOW TO ANSWER — every question resolves to exactly one of three states:

1. SUPPORTED — the excerpts contain the fact. Answer it.

2. FALSE BY CLOSURE — the question asks whether some team, player or event
   happened, the excerpts cover the full scope it asks about, and it does not
   appear there. The answer is then NO, and you must give it: say plainly
   that it did not happen and name the window you checked. Do NOT say the
   information is missing. A complete list that does not contain X is
   positive evidence that X did not occur within it. State the claim about
   the covered finals ("did not appear in any final between 2022 and 2026"),
   never about all of history.

3. OUTSIDE COVERAGE — the question needs a year, a competition, or a level of
   detail the corpus does not carry (see `unit` above: a per-series average,
   for instance, cannot come from a single game). Say what is covered, say
   what is missing, and give the closest fact you do hold.

Use ONLY the excerpts. Never use outside knowledge, and never infer facts
about matches the corpus does not cover.

ENTITY RULES
- A player listed in a team's lineup or box score played for that team in
  that final. If that team won, that player won that title — report it.
- "Won" can mean a team title or an individual award. When a question about a
  person is ambiguous between them, answer both senses.

SOURCE PRECEDENCE
- A source labelled "computed totals" is derived deterministically from
  structured data and outranks the prose excerpts. If they appear to
  disagree, follow the computed source silently.

OUTPUT CONTRACT
- Lead with the answer.
- Whenever you name a final, a season or a year, name the teams too.
- State the window you checked when answering existence, count, or "more than
  one" questions.
- Do not narrate your reasoning, your uncertainty, or the adequacy of the
  sources — except in state 3, where saying what is missing is the answer.

Today's date is {today}. Use it to resolve relative time references in the
question ("this year", "last year", "the last 3 years") — do not rely on your
own notion of the current date. Both competitions' finals are played in
May/June, so the season currently underway has no final to report yet."""

# Retrieval hands over every relevant final, but nothing in the text says so,
# so without this the model hedges ("among the available sources...") or
# answers from whichever excerpt it read first. `plan_evidence_ids` in
# src/retrieve.py is what makes the promise true.
COMPLETENESS_INSTRUCTION = """SCOPE OF THESE EXCERPTS
The excerpts below are the complete set of finals relevant to this question,
not a sample{window}. Base any count, ranking, superlative, comparison or
absence conclusion on all of them, and read every excerpt before naming a
maximum, a total, or something that never happened."""


def _window_phrase(plan: QueryPlan) -> str:
    if not plan.years:
        return ""
    years = sorted(plan.years)
    span = f"{years[0]}" if len(years) == 1 else f"{years[0]}-{years[-1]}"
    return f", covering {span}"


def build_prompt(question: str, chunks: list[dict], plan: QueryPlan | None = None) -> str:
    context = "\n\n".join(f"[Source: {c['source_title']} ({c['competition']} {c['season']})]\n{c['text']}" for c in chunks)
    sport = plan.sport_filter if plan else None
    system_prompt = SYSTEM_PROMPT.format(
        today=date.today().isoformat(),
        coverage=coverage_summary(sport),
    )

    if plan and plan.intent != "factoid":
        system_prompt = f"{system_prompt}\n\n{COMPLETENESS_INSTRUCTION.format(window=_window_phrase(plan))}"
        # Totals, margins and rankings arrive already computed in Python, so
        # the answer never depends on the model adding up a box score.
        computed = derived_facts_block(sport, plan.year_strings)
        if computed:
            context = f"[Source: computed totals over the finals below]\n{computed}\n\n{context}"

    return f"{system_prompt}\n\nSource excerpts:\n{context}\n\nQuestion (Hebrew): {question}\n\nAnswer (Hebrew):"


def generate_answer(question: str, chunks: list[dict], plan: QueryPlan | None = None) -> str:
    """Call Gemini to answer `question` in Hebrew, grounded only in `chunks`."""
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not set. Copy .env.example to .env and fill it in.")

    client = genai.Client(api_key=GEMINI_API_KEY)
    response = client.models.generate_content(
        model=GEMINI_MODEL_NAME,
        contents=build_prompt(question, chunks, plan),
    )
    return response.text.strip()
