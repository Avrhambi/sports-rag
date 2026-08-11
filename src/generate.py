"""Generate a grounded Hebrew answer from retrieved English chunks via Gemini."""

from datetime import date

from src import gemini
from src.config import coverage_summary
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

Describe the corpus ONLY from the block above. Never describe your coverage
by summarising which excerpts you happened to receive — the excerpts are a
selection, that block is the authority.

HOW TO ANSWER — every question resolves to exactly one of three cases. These
are internal categories: never print their names or numbers in your answer.

1. The excerpts contain the fact. Answer it.

2. The question asks whether some team, player or event happened, the
   excerpts cover the full scope it asks about, and it does not appear there.
   The answer is then NO, and you must give it: say plainly that it did not
   happen and name the window you checked. Do NOT say the information is
   missing. A complete list that does not contain X is positive evidence that
   X did not occur within it. State the claim about the covered finals ("did
   not appear in any final between 2022 and 2026"), never about all of
   history. If a set of values is described as all-distinct, that is the
   answer to "which occurs most often" — none of them does — not missing data.

3. The question needs a year, a competition, or a level of detail the corpus
   does not carry (see `unit` above: a per-series average, for instance,
   cannot come from a single game). Say what is covered, say what is missing,
   and give the closest fact you do hold.

Use ONLY the excerpts. Never use outside knowledge, and never infer facts
about matches the corpus does not cover. This includes names: if a source
gives only a surname, use only that surname — do not supply a first name
from your own knowledge.

ENTITY RULES
- A player listed in a team's lineup or box score played for that team in
  that final. If that team won, that player won that title — report it.
- "Won" can mean a team title or an individual award. Asked whether a person
  won, report BOTH: if they won a title, say so, and in the same answer name
  any individual award they took that year — and the reverse. Do not stop at
  whichever you find first. Where a source gives a person's whole record on
  one line, report all of it rather than its first clause.
- When a name could refer to more than one person in the covered finals (a
  player and a coach, say), answer for both and say which is which.

SOURCE PRECEDENCE
- A source labelled "computed totals" is derived deterministically from
  structured data and outranks the prose excerpts. If they appear to
  disagree, follow the computed source silently.

SCORELINES — the most common way this system has reported a match backwards
- Never write a bare "X–Y". Attach each number to its team: write
  "ריאל מדריד 1 – ליברפול 0", not "1–0".
- Prefer the pre-oriented "Score (winner first)" / "Deciding game (winner
  first)" column when one is present; it already names the winner first.
- Never reorder a scoreline you are copying. If you name the winner first,
  the winner's goals or points come first too.

OUTPUT CONTRACT
- Lead with the answer.
- Whenever you name a final, a season or a year, name the teams too.
- Put the window you checked BEFORE the number when answering existence,
  count, or "more than one" questions — "between 2022 and 2026 Real Madrid
  won twice", never "Real Madrid have 2 titles" followed by a qualifier. A
  count over the covered finals is not a career total, and the first clause
  is the part that gets read.
- Write plain Hebrew prose. No Markdown: no asterisks, no bullet syntax, no
  bold markers, no tables. Separate list items onto their own lines.
- Transliterate team and player names into Hebrew; give the Latin spelling in
  parentheses only where it genuinely helps.
- Do not narrate your reasoning, your uncertainty, or the adequacy of the
  sources — except in case 3, where saying what is missing is the answer.

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
        # Totals, margins and rankings arrive already computed in Python, so
        # the answer never depends on the model adding up a box score.
        computed = derived_facts_block(sport, plan.year_strings, plan.asks_about_person)
        if computed:
            # Only promise completeness when something was actually selected.
            # A plan for an uncovered year (2021, say) selects nothing, and
            # the prompt used to announce "the complete set of finals...
            # covering 2021" over an arbitrary top-4 -- inviting a fabricated
            # answer about a final the corpus has never held.
            system_prompt = f"{system_prompt}\n\n{COMPLETENESS_INSTRUCTION.format(window=_window_phrase(plan))}"
            context = f"[Source: computed totals over the finals below]\n{computed}\n\n{context}"
        else:
            system_prompt = (
                f"{system_prompt}\n\nSCOPE OF THESE EXCERPTS\nNo covered final matches the "
                "years this question asks about. The excerpts below are the nearest matches "
                "by similarity, not a complete set — do not count, rank or conclude an "
                "absence from them."
            )

    return f"{system_prompt}\n\nSource excerpts:\n{context}\n\nQuestion (Hebrew): {question}\n\nAnswer (Hebrew):"


def generate_answer(question: str, chunks: list[dict], plan: QueryPlan | None = None) -> str:
    """Call Gemini to answer `question` in Hebrew, grounded only in `chunks`."""
    if not gemini.has_key():
        raise RuntimeError(
            "No Gemini API key is set. Copy .env.example to .env and fill in GEMINI_API_KEY_1."
        )

    response = gemini.generate_content(build_prompt(question, chunks, plan))
    return response.text.strip()
