"""Fetch UEFA Champions League final data from Wikipedia wikitext and render
each final as a structured Markdown match report.

Wikipedia's "<year> UEFA Champions League final" pages follow a consistent
template: an infobox, a {{Football box}} in the "Details" section (date,
score, scorers with minutes, venue, referee), a two-column lineup table
(position, number, player, captain/card/substitution markers, manager), and
a "Statistics" section with team stat tables. This module parses that
wikitext directly via the MediaWiki API (no HTML scraping, no browser).
"""

import re

from src.config import UCL_FINALS_YEARS
from src.wikitext import clean_wikitext as _clean_generic
from src.wikitext import extract_section, fetch_wikitext

MONTHS = [
    "", "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]


def clean_wikitext(text: str) -> str:
    """Resolve football-specific templates, then hand off to the generic cleaner."""
    text = re.sub(r"\{\{flagicon\|[^}]*\}\}", "", text, flags=re.I)
    text = re.sub(r"\{\{fbaicon\|[^}]*\}\}", "", text, flags=re.I)
    text = re.sub(r"\{\{fba\|[^}]*\}\}", "", text, flags=re.I)
    text = re.sub(r"\{\{#invoke:flag\|[^}]*\}\}", "", text, flags=re.I)
    text = re.sub(r"\{\{nowrap\|([^}]*)\}\}", r"\1", text, flags=re.I)
    text = re.sub(r"\{\{goal\|([^}|]+)(?:\|[^}]*)*\}\}", r"\1'", text, flags=re.I)
    text = re.sub(r"\{\{yel\|([^}|]*)\}\}", r"Yellow card \1'", text, flags=re.I)
    text = re.sub(r"\{\{red\|([^}|]*)\}\}", r"Red card \1'", text, flags=re.I)
    text = re.sub(r"\{\{suboff\|([^}|]*)\}\}", r"Substituted off \1'", text, flags=re.I)
    text = re.sub(r"\{\{subon\|([^}|]*)\}\}", r"Substituted on \1'", text, flags=re.I)
    return _clean_generic(text)


def parse_start_date(raw: str) -> str:
    match = re.search(r"\{\{Start date\|(\d+)\|(\d+)\|(\d+)", raw)
    if not match:
        return clean_wikitext(raw)
    year, month, day = (int(x) for x in match.groups())
    return f"{day} {MONTHS[month]} {year}"


def parse_football_box(details_text: str) -> dict:
    """Parse the football-box template (older `{{Football box`, newer
    `{{#invoke:football box|main`) into a field dict."""
    start_match = re.search(r"\{\{(?:Football box|#invoke:football box\|main)", details_text, re.I)
    if not start_match:
        raise ValueError("No football-box template found")
    start = start_match.start()
    depth = 0
    end = start
    for i in range(start, len(details_text)):
        if details_text[i : i + 2] == "{{":
            depth += 1
        elif details_text[i : i + 2] == "}}":
            depth -= 1
            if depth == 0:
                end = i + 2
                break
    block = details_text[start:end]

    fields: dict[str, str] = {}
    current_key = None
    for line in block.splitlines():
        field_match = re.match(r"\|\s*(\w+)\s*=(.*)$", line)
        if field_match:
            current_key = field_match.group(1)
            fields[current_key] = field_match.group(2)
        elif current_key is not None:
            fields[current_key] += "\n" + line

    return fields


def parse_goals(raw: str) -> list[str]:
    entries = [line.lstrip("*").strip() for line in raw.splitlines() if line.strip().startswith("*")]
    return [clean_wikitext(e) for e in entries if clean_wikitext(e)]


def parse_penalties(raw: str) -> list[dict]:
    """Parse a penalties1/penalties2 field into [{player, scored}, ...], in kick order."""
    entries = []
    for line in raw.splitlines():
        line = line.strip()
        if not line.startswith("*"):
            continue
        link_match = re.search(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]", line)
        if not link_match:
            continue
        player = clean_wikitext(link_match.group(1))
        scored = bool(re.search(r"\{\{pengoal\}\}", line, re.I))
        entries.append({"player": player, "scored": scored})
    return entries


def parse_lineup_column(column_text: str) -> dict:
    """Parse one team's lineup cell: starting XI, substitutes used, manager."""
    starters, substitutes, manager = [], [], ""
    seen_subs_marker = False
    for line in column_text.splitlines():
        line = line.strip()
        if not line.startswith("|") or line.startswith("|-") or line.startswith("|}"):
            continue
        if "Substitutes:" in line:
            seen_subs_marker = True
            continue
        if "Manager:" in line:
            continue
        cells = [re.sub(r"^colspan=\d+\|", "", c.strip()) for c in line.lstrip("|").split("||")]
        if len(cells) < 3:
            # Likely the manager's own row (colspan=3|{{flagicon|..}} [[Name]])
            if cells and manager == "" and starters and seen_subs_marker:
                candidate = clean_wikitext(cells[0])
                if candidate:
                    manager = candidate
            continue
        position, number, player_cell = cells[0], cells[1], cells[2]
        events = [clean_wikitext(c) for c in cells[3:] if clean_wikitext(c)]
        entry = {
            "position": clean_wikitext(position),
            "number": clean_wikitext(number),
            "player": clean_wikitext(player_cell),
            "events": events,
        }
        if not entry["player"]:
            continue
        (substitutes if seen_subs_marker else starters).append(entry)
    return {"starters": starters, "substitutes": substitutes, "manager": manager}


def parse_lineups(details_text: str) -> dict:
    """Split the details section's lineup table into team1/team2 columns."""
    parts = re.split(r'\|valign="top"[^\n|]*\|', details_text)
    if len(parts) < 4:
        raise ValueError("Could not locate two-column lineup table")
    team1 = parse_lineup_column(parts[1])
    team2 = parse_lineup_column(parts[3])
    return {"team1": team1, "team2": team2}


def parse_statistics(wikitext: str) -> dict | None:
    try:
        stats_section = extract_section(wikitext, "Statistics")
    except ValueError:
        return None
    overall_match = re.search(r"\|\+Overall.*?(?=\{\{col-end\}\}|\Z)", stats_section, re.S)
    if not overall_match:
        return None
    block = overall_match.group()
    cols = re.findall(r'!scope="col"[^\n|]*\|([^\n]+)', block)
    if len(cols) < 3:
        return None
    team1_name, team2_name = clean_wikitext(cols[1]), clean_wikitext(cols[2])
    rows = re.findall(r"!scope=row\|([^\n]+)\n\|([^\n]+)\n\|([^\n]+)", block)
    stats = [
        {"label": clean_wikitext(label), "team1": clean_wikitext(v1), "team2": clean_wikitext(v2)}
        for label, v1, v2 in rows
    ]
    return {"team1_name": team1_name, "team2_name": team2_name, "rows": stats}


def parse_final(wikitext: str) -> dict:
    details = extract_section(wikitext, "Details")
    box = parse_football_box(details)
    lineups = parse_lineups(details)
    stats = parse_statistics(wikitext)

    team1 = clean_wikitext(box.get("team1", ""))
    team2 = clean_wikitext(box.get("team2", ""))
    score = clean_wikitext(box.get("score", ""))

    event_match = re.search(r"\|\s*event\s*=(.*)", wikitext[: wikitext.index("==Background==")] if "==Background==" in wikitext else wikitext)
    season = ""
    if event_match:
        cleaned_event = clean_wikitext(event_match.group(1))
        season_match = re.search(r"(\d{4}–\d{2,4})", cleaned_event)
        if season_match:
            season = season_match.group(1)

    return {
        "team1": team1,
        "team2": team2,
        "score": score,
        "date": parse_start_date(box.get("date", "")),
        "stadium": clean_wikitext(box.get("stadium", "")),
        "attendance": clean_wikitext(box.get("attendance", "")),
        "referee": clean_wikitext(box.get("referee", "")),
        "goals1": parse_goals(box.get("goals1", "")),
        "goals2": parse_goals(box.get("goals2", "")),
        "after_extra_time": clean_wikitext(box.get("aet", "")).lower() == "yes",
        "penalty_score": clean_wikitext(box.get("penaltyscore", "")),
        "penalties1": parse_penalties(box.get("penalties1", "")),
        "penalties2": parse_penalties(box.get("penalties2", "")),
        "season": season,
        "lineups": lineups,
        "statistics": stats,
    }


def render_markdown(year: int, match: dict) -> str:
    lines = [f"# {year} UEFA Champions League Final (Football / Soccer)", ""]
    lines += ["## Match Info"]
    lines += [
        f"- Competition: UEFA Champions League{f' ({match['season']})' if match['season'] else ''}",
        f"- Date: {match['date']}",
        f"- Venue: {match['stadium']}",
        f"- Attendance: {match['attendance']}",
        f"- Referee: {match['referee']}",
        f"- Result: {match['team1']} {match['score']} {match['team2']}"
        + (" (after extra time)" if match["after_extra_time"] else ""),
    ]
    if match["penalty_score"]:
        lines.append(f"- Penalty shootout: {match['team1']} {match['penalty_score']} {match['team2']}")
    lines.append("")

    lines += ["## Lineups"]
    for team_key, team_name in (("team1", match["team1"]), ("team2", match["team2"])):
        team = match["lineups"][team_key]
        lines.append(f"### {team_name} (Manager: {team['manager']})")
        starters = "; ".join(
            f"{p['position']} #{p['number']} {p['player']}" + (f" ({', '.join(p['events'])})" if p["events"] else "")
            for p in team["starters"]
        )
        lines.append(f"Starting XI: {starters}")
        subs = "; ".join(
            f"#{p['number']} {p['player']}" + (f" ({', '.join(p['events'])})" if p["events"] else "")
            for p in team["substitutes"]
        )
        if subs:
            lines.append(f"Substitutes: {subs}")
        lines.append("")

    lines += ["## Key Events"]
    for goal in match["goals1"]:
        lines.append(f"- Goal ({match['team1']}): {goal}")
    for goal in match["goals2"]:
        lines.append(f"- Goal ({match['team2']}): {goal}")
    for team_key, team_name in (("team1", match["team1"]), ("team2", match["team2"])):
        team = match["lineups"][team_key]
        for p in team["starters"] + team["substitutes"]:
            for event in p["events"]:
                lines.append(f"- {event} ({team_name}): {p['player']}")
    if match["penalties1"] or match["penalties2"]:
        lines.append(f"- Penalty shootout ({match['team1']} vs {match['team2']}), in kick order:")
        for team_name, kicks in ((match["team1"], match["penalties1"]), (match["team2"], match["penalties2"])):
            for kick in kicks:
                outcome = "scored" if kick["scored"] else "missed"
                lines.append(f"  - {team_name}: {kick['player']} {outcome}")
    lines.append("")

    if match["statistics"]:
        stats = match["statistics"]
        lines += ["## Statistics", f"| Statistic | {stats['team1_name']} | {stats['team2_name']} |", "|---|---|---|"]
        for row in stats["rows"]:
            lines.append(f"| {row['label']} | {row['team1']} | {row['team2']} |")
        lines.append("")

    return "\n".join(lines)


def build_football_docs() -> list[dict]:
    """Fetch and render every configured UCL final as a Markdown match report."""
    docs = []
    for year in UCL_FINALS_YEARS:
        title = f"{year} UEFA Champions League final"
        try:
            wikitext = fetch_wikitext(title)
            match = parse_final(wikitext)
            markdown = render_markdown(year, match)
        except Exception as exc:  # noqa: BLE001 - one bad year shouldn't kill the whole ingest
            print(f"[ingest_football] skipping {title}: {exc}")
            continue
        docs.append(
            {
                "markdown": markdown,
                "sport": "football",
                "competition": "UEFA Champions League",
                "season": match["season"] or str(year),
                "teams": f"{match['team1']} vs {match['team2']}",
                "source_title": title,
                "url": "https://en.wikipedia.org/wiki/" + title.replace(" ", "_"),
            }
        )
    return docs


if __name__ == "__main__":
    built_docs = build_football_docs()
    print(f"Built {len(built_docs)} UCL final match reports.")
    for d in built_docs:
        print(f"- {d['source_title']}: {d['teams']}")
