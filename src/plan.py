"""Turn a Hebrew question into a structured retrieval plan.

Retrieval needs three things the query text doesn't hand over directly: which
sport to filter to, which finals' reports must be in context, and whether the
question is answerable from a single top-scoring chunk at all. A Gemini call
extracts all three at once.

This replaces a growing pile of hand-written Hebrew regexes. Those regexes
survive below as `heuristic_plan`, the offline fallback used when no API key
is configured (the eval's retrieval tier is meant to run without one) or when
the planner call fails -- but they are frozen: new phrasings are the
planner's job, not another regex.

Run as: python -m src.plan "<question>"
"""

import re
from datetime import date
from functools import lru_cache
from typing import Literal

from pydantic import BaseModel

from src import gemini
from src.config import NBA_FINALS_YEARS, UCL_FINALS_YEARS

Sport = Literal["football", "basketball", "any"]
Intent = Literal["factoid", "existence", "aggregate", "comparison", "multihop"]


class QueryPlan(BaseModel):
    """What retrieval needs to know about a question before searching."""

    sport: Sport = "any"
    # Every year whose match report is needed. Empty means "no year
    # constraint" -- rank by similarity alone.
    years: list[int] = []
    intent: Intent = "factoid"
    # True when the planner was unavailable and this came from the frozen
    # regex fallback. That path always reports `factoid`, which silently
    # switches off the completeness promise, the computed facts block and
    # whole-sport evidence -- i.e. every closed-world fix in the pipeline.
    # Callers surface it rather than answering as if nothing had changed.
    degraded: bool = False

    @property
    def sport_filter(self) -> str | None:
        """The sport to filter/boost by, or None to leave both in play."""
        return None if self.sport == "any" else self.sport

    @property
    def year_strings(self) -> set[str]:
        """Years as strings, for matching against chunk source titles."""
        return {str(year) for year in self.years}


PLANNER_PROMPT = """You plan retrieval for a Hebrew-language sports-history
question answering system. The corpus holds one match report per final, and
covers only these finals:
- football: the UEFA Champions League final of {ucl_years}
- basketball: the NBA Finals clinching game of {nba_years}

Decide three things about the question:

sport: "football" if it is about the Champions League, "basketball" if it is
about the NBA Finals, or "any" if it spans both or names neither.

years: every year whose match report is needed to answer. Use an empty list
if no particular year is implied. A superlative or a count over the whole
corpus ("which final had the biggest margin", "how many times did X win")
needs EVERY covered year of that sport listed, not just one. Never return a
year outside the covered lists above.

intent:
- "factoid": one fact from one match (winner, venue, referee, MVP, score).
- "existence": a yes/no question about whether some team, player or event
  ever appears ("did Barcelona win", "did X not win", "did X win more than
  once"). Answering "no" needs the whole covered set, so list every covered
  year of the relevant sport, exactly as for an aggregate.
- "aggregate": counting, summing, or a superlative across several matches
  ("how many times", "who scored the most", "the biggest margin").
- "comparison": facts from several matches set side by side.
- "multihop": one match's fact decides which other match to look up (e.g.
  "who coached the NBA champion in the year team X won the Champions League").

Today is {today}. Both competitions' finals are played in May or June, so the
season currently underway has no final yet. Resolve relative expressions
("השנה", "בשנה שעברה", "בשלוש השנים האחרונות", "לאחרונה") to concrete years
against that rule rather than to your own notion of the present.

Question (Hebrew): {question}"""


# --- Offline fallback (frozen -- extend the planner prompt instead) ---------

YEAR_PATTERN = re.compile(r"\b(?:19|20)\d{2}\b")

# Hebrew number words for "the last N years" phrasing, e.g. "בשלוש השנים האחרונות".
_HEBREW_YEAR_COUNT_WORDS = {
    "שלוש": 3, "שלושה": 3,
    "ארבע": 4, "ארבעה": 4,
    "חמש": 5, "חמישה": 5,
}
_LAST_N_YEARS_DIGIT = re.compile(r"(\d+)\s*ה?שנים\s*ה?אחרונות")
_LAST_N_YEARS_WORD = re.compile(
    r"(" + "|".join(_HEBREW_YEAR_COUNT_WORDS) + r")\s*ה?שנים\s*ה?אחרונות"
)
_LAST_TWO_YEARS = re.compile(r"שנתיים\s*ה?אחרונות")
_THIS_YEAR = re.compile(r"\bהשנה\b")
_LAST_YEAR = re.compile(r"בשנה שעברה|שנה שעברה|אשתקד")
# No explicit count ("in recent years", "recently") -- default to a
# generous window since the whole seeded corpus is only 5 years deep.
_RECENT_YEARS_NO_COUNT = re.compile(r"בשנים\s*ה?אחרונות|לאחרונה")
_RECENT_YEARS_DEFAULT_N = 5

# Competition-identifying keywords are a much more reliable sport signal than
# raw cosine similarity for this small, closed two-sport domain.
SPORT_KEYWORDS = {
    "football": ["כדורגל", "ליגת האלופות", "champions league", "uefa", "football", "soccer"],
    "basketball": ["כדורסל", "פיינלס", "nba", "basketball"],
}


def most_recent_completed_final_year(today: date | None = None) -> int:
    """Both UCL and NBA finals are played in May/June. From July onward the
    next season is already underway and this calendar year's final has
    already happened; before July, last calendar year's final is the most
    recently completed one (a safe approximation around the fuzzy
    May/June boundary -- exact playoff calendars vary slightly by year)."""
    today = today or date.today()
    return today.year if today.month >= 7 else today.year - 1


def resolve_relative_years(query: str, anchor_year: int, today: date | None = None) -> set[str]:
    """Translate Hebrew relative-year phrases ("השנה", "שנה שעברה", "בשלוש
    השנים האחרונות") into absolute year strings anchored to `anchor_year`
    (the most recently completed final).

    "This year" and "last year" depend on whether this calendar year's final
    has already been played -- from July onward it has. The original mapping
    assumed it had not, so after July it ran a year late: in August 2026 it
    resolved "השנה" to 2027, a year the corpus cannot hold, and "שנה שעברה"
    to 2026 rather than 2025."""
    years: set[str] = set()
    played_this_year = anchor_year == (today or date.today()).year

    if _THIS_YEAR.search(query):
        # Before July there is no final this calendar year yet; the most
        # recently completed one is the closest thing to what was asked.
        years.add(str(anchor_year))
    if _LAST_YEAR.search(query):
        years.add(str(anchor_year - 1 if played_this_year else anchor_year))
    if _LAST_TWO_YEARS.search(query):
        years.update(str(y) for y in range(anchor_year - 1, anchor_year + 1))

    word_match = _LAST_N_YEARS_WORD.search(query)
    if word_match:
        n = _HEBREW_YEAR_COUNT_WORDS[word_match.group(1)]
        years.update(str(y) for y in range(anchor_year - n + 1, anchor_year + 1))

    digit_match = _LAST_N_YEARS_DIGIT.search(query)
    if digit_match:
        n = int(digit_match.group(1))
        years.update(str(y) for y in range(anchor_year - n + 1, anchor_year + 1))

    if not years and _RECENT_YEARS_NO_COUNT.search(query):
        n = _RECENT_YEARS_DEFAULT_N
        years.update(str(y) for y in range(anchor_year - n + 1, anchor_year + 1))

    return years


def detect_sport(query: str) -> Sport:
    """Best-effort sport guess from competition-name keywords in the query."""
    lowered = query.lower()
    for sport_name, keywords in SPORT_KEYWORDS.items():
        if any(keyword in lowered for keyword in keywords):
            return sport_name  # type: ignore[return-value]
    return "any"


def heuristic_plan(question: str) -> QueryPlan:
    """Keyword/regex plan for when the planner is unavailable. Always reports
    `factoid` intent -- guessing at aggregation from surface patterns is what
    the planner exists to avoid -- so callers degrade to plain top-k."""
    years = set(YEAR_PATTERN.findall(question))
    years |= resolve_relative_years(question, most_recent_completed_final_year())
    return QueryPlan(
        sport=detect_sport(question),
        years=sorted(int(y) for y in years),
        degraded=True,
    )


# --- Planner ---------------------------------------------------------------


@lru_cache(maxsize=256)
def plan_query(question: str) -> QueryPlan:
    """Plan retrieval for `question`, falling back to `heuristic_plan` when
    Gemini is unavailable. Cached: the eval and the API can both ask the same
    question more than once, and the plan only depends on the question."""
    if not gemini.has_key():
        return heuristic_plan(question)

    prompt = PLANNER_PROMPT.format(
        ucl_years=", ".join(str(y) for y in UCL_FINALS_YEARS),
        nba_years=", ".join(str(y) for y in NBA_FINALS_YEARS),
        today=date.today().isoformat(),
        question=question,
    )
    try:
        response = gemini.generate_content(
            prompt,
            config={"response_mime_type": "application/json", "response_schema": QueryPlan},
        )
    except Exception as exc:  # noqa: BLE001 - a planner outage must not break search
        print(f"[plan] falling back to heuristics: {exc}")
        return heuristic_plan(question)

    plan = response.parsed
    if not isinstance(plan, QueryPlan):
        print("[plan] falling back to heuristics: planner returned no usable plan")
        return heuristic_plan(question)
    return plan


if __name__ == "__main__":
    import sys

    query_arg = sys.argv[1] if len(sys.argv) > 1 else "כמה פעמים ריאל מדריד זכתה בשנים האחרונות?"
    print(plan_query(query_arg))
