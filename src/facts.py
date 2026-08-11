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


def _tally_with_years(finals: list[dict], key: str, noun: str) -> str:
    """`Team 2 (2022, 2024)` -- the count and the years that make it up. A
    bare count forced the model to go looking for which years, and it guessed
    wrong (it once put Boston among the 2024 losers, the year Boston won)."""
    by_value: defaultdict[str, list[int]] = defaultdict(list)
    unknown = 0
    for f in finals:
        value = f.get(key)
        if value:
            by_value[value].append(f["year"])
        else:
            unknown += 1
    if not by_value:
        return ""
    ranked = sorted(by_value.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    body = "; ".join(f"{name} {len(years)} ({', '.join(str(y) for y in sorted(years))})" for name, years in ranked)
    # An unparsed row would otherwise vanish and the undercount would read as a fact.
    suffix = f" [{unknown} final(s) with no recorded {key}]" if unknown else ""
    return f"{noun}: {body}{suffix}"


def _distinctness(finals: list[dict], key: str, noun: str) -> str:
    """State explicitly when every value is different. Without this the model
    read "no maximum in the computed block" as "the data is missing", declared
    it could not answer, and then printed all five values."""
    counts = Counter(f[key] for f in finals if f.get(key))
    if not counts:
        return ""
    repeats = [f"{name} ({n})" for name, n in counts.most_common() if n > 1]
    if repeats:
        return f"{noun} appearing more than once: {', '.join(repeats)}"
    return f"{noun}: all {len(counts)} distinct — none appears more than once, so there is no maximum"


def _individual_honours(finals: list[dict]) -> list[str]:
    """One row per person, keyed by the person.

    Pre-joining is not enough on its own -- it has to be pre-joined in the
    direction the question asks. A per-year squad list answers "who was in the
    2024 squad"; asked "did Jayson Tatum win", the model scanned for the most
    answer-shaped row it could find, hit the MVP column, and returned a
    confident "no" while his name sat inside a comma-separated squad a few
    lines above. Keying by person makes that question a lookup.

    Each row also states its negatives outright -- "no title", "no Finals
    MVP" -- rather than leaving them as a missing clause. A row that simply
    omitted the award produced a hedge ("it isn't known whether he won
    individual awards") for a fact the corpus settles, which is the same
    failure as reading an absent maximum as absent data. A negative the model
    can copy beats one it has to notice.
    """
    honours: defaultdict[str, dict] = defaultdict(lambda: {"titles": [], "lost": [], "mvp": [], "roles": set()})
    for final in finals:
        for person in final.get("people", []):
            record = honours[person["name"]]
            record["roles"].add(person["role"])
            bucket = "titles" if person["team_result"] == "won" else "lost"
            record[bucket].append((final["year"], person["team"]))
        if final.get("mvp"):
            honours[final["mvp"]]["mvp"].append(final["year"])

    if not honours:
        return []

    # Only football finals have no MVP in this corpus; stamping "no Finals
    # MVP" on a football squad would assert something never modelled.
    tracks_awards = any(final.get("mvp") for final in finals)

    header = (
        "Individual record (one row per person; a person absent from this list "
        "did not take part in any of these finals):"
    )
    lines = [header]
    for name in sorted(honours):
        record = honours[name]
        parts = []
        if record["titles"]:
            parts.append("won " + ", ".join(f"{y} with {t}" for y, t in sorted(record["titles"])))
        else:
            parts.append("no title")
        if record["lost"]:
            parts.append("lost " + ", ".join(f"{y} with {t}" for y, t in sorted(record["lost"])))
        if tracks_awards:
            parts.append(
                "Finals MVP " + ", ".join(str(y) for y in sorted(record["mvp"]))
                if record["mvp"]
                else "no Finals MVP"
            )
        role = "/".join(sorted(record["roles"])) or "player"
        lines.append(f"- {name} ({role}): {'; '.join(parts)}")
    return lines


def _winning_squads(finals: list[dict]) -> list[str]:
    """Every person on a winning side, listed under their year. Answers "did
    this player win?" by reading rather than by joining a roster chunk to a
    result chunk -- the inference that produced confident wrong negatives."""
    lines = ["Championship-winning squads (every person named here won that year's title):"]
    for f in finals:
        winners = [p for p in f.get("people", []) if p["team_result"] == "won"]
        if not winners:
            continue
        names = ", ".join(
            f"{p['name']} ({p['role']})" if p["role"] not in ("player", "starter") else p["name"]
            for p in winners
        )
        lines.append(f"- {f['year']} {f['winner']}: {names}")
    lines.append(
        "Anyone not named above did not win one of these finals. Anyone not named "
        "anywhere in these reports did not appear in them at all."
    )
    return lines


def football_block(finals: list[dict]) -> list[str]:
    lines = ["### UEFA Champions League finals (computed from structured data)"]
    lines.append("| Year | Winner | Runner-up | Score (winner first) | Margin | Attendance | Venue | Referee |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for f in finals:
        lines.append(
            f"| {f['year']} | {f['winner'] or '-'} | {f['loser'] or '-'} | "
            f"{f.get('score_winner_first') or f['score']} | "
            f"{f['goal_margin'] if f['goal_margin'] is not None else '-'} | "
            f"{f['attendance'] if f['attendance'] is not None else '-'} | "
            f"{f.get('venue') or '-'} | {f.get('referee') or '-'} |"
        )

    lines.append("")
    lines.append(_tally_with_years(finals, "winner", "Titles won"))
    lines.append(_tally_with_years(finals, "loser", "Finals lost"))

    appearances: defaultdict[str, list[int]] = defaultdict(list)
    for f in finals:
        for team in {f["team1"], f["team2"]}:
            appearances[team].append(f["year"])
    repeats = [f"{t} ({', '.join(str(y) for y in sorted(ys))})" for t, ys in appearances.items() if len(ys) > 1]
    lines.append(f"Teams in more than one final: {', '.join(repeats) if repeats else 'none'}")

    with_margin = [f for f in finals if f["goal_margin"] is not None]
    if with_margin:
        widest = max(with_margin, key=lambda f: f["goal_margin"])
        lines.append(
            f"Largest goal margin: {widest['year']}, {widest['goal_margin']} goals "
            f"({widest.get('score_winner_first') or widest['score']})"
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

    lines.append(_distinctness(finals, "referee", "Referees"))
    lines.append(_distinctness(finals, "venue", "Venues"))
    lines.append("")
    lines += _winning_squads(finals)
    lines.append("")
    lines += _individual_honours(finals)
    return lines


def basketball_block(finals: list[dict]) -> list[str]:
    lines = ["### NBA Finals (computed from structured data)"]
    lines.append("| Year | Champion | Runner-up | Series | Deciding game (winner first) | Point margin | MVP | Venue |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for f in finals:
        game = f["deciding_game"]
        score = game.get("score_winner_first") or (
            f"{game['away_team']} {game['away_score']} – {game['home_team']} {game['home_score']}"
        )
        lines.append(
            f"| {f['year']} | {f['champion']} | {f['runnerup']} | {f['series_score'] or '-'} | "
            f"{score} | {game['point_margin']} | {f['mvp']} | {f.get('venue') or '-'} |"
        )

    lines.append("")
    lines.append(_tally_with_years(finals, "champion", "Titles won"))
    lines.append(_tally_with_years(finals, "runnerup", "Finals lost"))

    # Pre-joined so a "which series went N games" question is read off a row
    # rather than assembled from a Series column and a Champion column.
    lines.append("Series length, with the teams:")
    for f in sorted(finals, key=lambda f: f["year"]):
        if f["games_played"]:
            lines.append(
                f"- {f['year']}: {f['games_played']} games — {f['champion']} beat "
                f"{f['runnerup']} {f['series_score']}"
            )

    longest = [f for f in finals if f["games_played"]]
    if longest:
        most = max(longest, key=lambda f: f["games_played"])
        lines.append(
            f"Longest series: {most['year']} ({most['games_played']} games, {most['champion']} "
            f"beat {most['runnerup']} {most['series_score']})"
        )

    widest = max(finals, key=lambda f: f["deciding_game"]["point_margin"])
    lines.append(
        f"Largest deciding-game margin: {widest['year']}, {widest['deciding_game']['point_margin']} "
        f"points ({widest['deciding_game'].get('score_winner_first')})"
    )

    lines.append(_distinctness(finals, "venue", "Venues"))
    lines.append(_tally_with_years(finals, "mvp", "Finals MVP awards"))

    # Deciding games only -- the corpus holds no other game of each series.
    #
    # Single-game highs and multi-year sums sit one line apart and are trivial
    # to confuse: an eval run picked "Jaylen Brown 55" (two games added) as the
    # answer to "most points in a game", which is Jalen Brunson's 45. Hence the
    # explicit headings, and the addends spelled out beside every sum -- a
    # figure that shows its working can't be mistaken for a single-game score.
    best_single = []
    per_player: defaultdict[str, list[tuple[int, int]]] = defaultdict(list)
    for f in finals:
        if not f["players"]:
            continue
        top = max(f["players"], key=lambda p: p["points"])
        best_single.append((top["points"], top["name"], f["year"]))
        for p in f["players"]:
            per_player[p["name"]].append((f["year"], p["points"]))

    if not best_single:
        return lines

    lines.append("")
    lines.append("Single-game scoring (one player in one deciding game):")
    for points, name, year in sorted(best_single, key=lambda b: b[2]):
        lines.append(f"- {year} deciding game, top scorer: {name} {points}")
    points, name, year = max(best_single)
    lines.append(f"- Most points by one player in a single deciding game: {name}, {points} ({year})")

    lines.append(
        "Multi-year sums (one player's points added across the deciding games "
        "above -- a total over several years, never a single-game score):"
    )
    # No rank cap: truncating to a top five made a player outside it look like
    # a coverage gap, and the model reported missing data for a question the
    # box scores answer in full. At ten finals the whole table is a few dozen
    # lines, so completeness costs less than the hedging did.
    totals = sorted(per_player.items(), key=lambda kv: sum(p for _, p in kv[1]), reverse=True)
    for name, games in totals:
        addends = " + ".join(f"{year}: {points}" for year, points in sorted(games))
        lines.append(f"- {name}: {sum(p for _, p in games)} total ({addends})")
    lines.append("This list covers every player in these box scores; there are no others.")

    lines.append("")
    lines += _winning_squads(finals)
    lines.append("")
    lines += _individual_honours(finals)
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
