"""Deterministic aggregates over the structured per-final records.

Counting, ranking and summing are the operations a language model is least
reliable at and Python is most reliable at, so they happen here and the
result is handed to the model as another source. The model still writes the
answer; it just doesn't have to do the arithmetic.

This is deliberately not a query engine. The corpus is ten finals, so every
aggregate worth asking about fits in a block small enough to hand over whole
-- no SQL, no query planning, nothing to get wrong at answer time.

Run as: python -m src.facts [football|basketball]
"""

import json
from collections import Counter, defaultdict
from functools import lru_cache

from src.config import FACTS_PATH


@lru_cache(maxsize=1)
def load_facts() -> list[dict]:
    """Every final's structured record, or an empty list before `src.ingest`
    has been run (callers degrade to prose-only context)."""
    if not FACTS_PATH.exists():
        return []
    return json.loads(FACTS_PATH.read_text(encoding="utf-8"))


def select(sport: str | None = None, years: set[str] | None = None) -> list[dict]:
    """The records a plan's sport/year filters allow, oldest first."""
    selected = [
        f
        for f in load_facts()
        if (not sport or f["sport"] == sport) and (not years or str(f["year"]) in years)
    ]
    return sorted(selected, key=lambda f: f["year"])


def _tally(counts: Counter, noun: str) -> str:
    ranked = ", ".join(f"{name} {n}" for name, n in counts.most_common() if name)
    return f"{noun}: {ranked}" if ranked else ""


def football_block(finals: list[dict]) -> list[str]:
    lines = ["### UEFA Champions League finals (computed from structured data)"]
    lines.append("| Year | Result | Winner | Goal margin | Attendance |")
    lines.append("|---|---|---|---|---|")
    for f in finals:
        result = f"{f['team1']} {f['score']} {f['team2']}"
        if f["decided_on_penalties"]:
            result += f" (penalties {f['penalty_score']})"
        lines.append(
            f"| {f['year']} | {result} | {f['winner'] or '-'} | "
            f"{f['goal_margin'] if f['goal_margin'] is not None else '-'} | "
            f"{f['attendance'] if f['attendance'] is not None else '-'} |"
        )

    lines.append("")
    lines.append(_tally(Counter(f["winner"] for f in finals), "Titles won"))
    lines.append(_tally(Counter(f["loser"] for f in finals), "Finals lost"))

    appearances = Counter()
    for f in finals:
        appearances.update({f["team1"], f["team2"]})
    repeats = [f"{team} {n}" for team, n in appearances.most_common() if n > 1]
    lines.append(f"Teams in more than one final: {', '.join(repeats) if repeats else 'none'}")

    with_margin = [f for f in finals if f["goal_margin"] is not None]
    if with_margin:
        widest = max(with_margin, key=lambda f: f["goal_margin"])
        lines.append(
            f"Largest goal margin: {widest['year']} ({widest['goal_margin']}, "
            f"{widest['team1']} {widest['score']} {widest['team2']})"
        )

    shootouts = [str(f["year"]) for f in finals if f["decided_on_penalties"]]
    lines.append(f"Decided on penalties: {', '.join(shootouts) if shootouts else 'none'}")

    with_crowd = [f for f in finals if f["attendance"] is not None]
    if with_crowd:
        biggest = max(with_crowd, key=lambda f: f["attendance"])
        smallest = min(with_crowd, key=lambda f: f["attendance"])
        lines.append(
            f"Highest attendance: {biggest['year']} ({biggest['attendance']:,} at {biggest['venue']}); "
            f"lowest: {smallest['year']} ({smallest['attendance']:,})"
        )
    return lines


def basketball_block(finals: list[dict]) -> list[str]:
    lines = ["### NBA Finals (computed from structured data)"]
    lines.append("| Year | Champion | Series | Deciding game | Point margin | MVP |")
    lines.append("|---|---|---|---|---|---|")
    for f in finals:
        game = f["deciding_game"]
        score = (
            f"{game['away_team']} {game['away_score']} – "
            f"{game['home_team']} {game['home_score']}"
        )
        lines.append(
            f"| {f['year']} | {f['champion']} | {f['series_score'] or '-'} | {score} | "
            f"{game['point_margin']} | {f['mvp']} |"
        )

    lines.append("")
    lines.append(_tally(Counter(f["champion"] for f in finals), "Titles won"))
    lines.append(_tally(Counter(f["runnerup"] for f in finals), "Finals lost"))

    longest = [f for f in finals if f["games_played"]]
    if longest:
        most = max(longest, key=lambda f: f["games_played"])
        lines.append(
            f"Longest series: {most['year']} ({most['games_played']} games, {most['series_score']})"
        )

    widest = max(finals, key=lambda f: f["deciding_game"]["point_margin"])
    lines.append(
        f"Largest deciding-game margin: {widest['year']} "
        f"({widest['deciding_game']['point_margin']} points)"
    )

    # Deciding games only -- the corpus holds no other game of each series, so
    # these are per-game highs and career totals over those games alone.
    best_single = []
    career_points: defaultdict[str, int] = defaultdict(int)
    for f in finals:
        if not f["players"]:
            continue
        top = max(f["players"], key=lambda p: p["points"])
        best_single.append((top["points"], top["name"], f["year"]))
        for p in f["players"]:
            career_points[p["name"]] += p["points"]

    if best_single:
        points, name, year = max(best_single)
        lines.append(f"Highest individual score in a deciding game: {name}, {points} points ({year})")
        per_year = "; ".join(f"{y}: {n} {pts}" for pts, n, y in sorted(best_single, key=lambda b: b[2]))
        lines.append(f"Top scorer per deciding game: {per_year}")
        totals = sorted(career_points.items(), key=lambda kv: kv[1], reverse=True)[:5]
        lines.append(
            "Most points across all deciding games in range: "
            + ", ".join(f"{n} {p}" for n, p in totals)
        )
    return lines


def derived_facts_block(sport: str | None = None, years: set[str] | None = None) -> str:
    """A Markdown block of pre-computed aggregates for the selected finals,
    or "" when there is nothing to compute over."""
    finals = select(sport, years)
    if not finals:
        return ""

    blocks: list[list[str]] = []
    football = [f for f in finals if f["sport"] == "football"]
    basketball = [f for f in finals if f["sport"] == "basketball"]
    if football:
        blocks.append(football_block(football))
    if basketball:
        blocks.append(basketball_block(basketball))

    lines = [line for block in blocks for line in [*block, ""]]
    return "\n".join(line for line in lines if line is not None).strip()


if __name__ == "__main__":
    import sys

    sport_arg = sys.argv[1] if len(sys.argv) > 1 else None
    print(derived_facts_block(sport_arg) or "No facts found - run `python -m src.ingest` first.")
