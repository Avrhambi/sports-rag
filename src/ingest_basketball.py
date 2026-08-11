"""Fetch NBA Finals data and render each series' clinching game as a
structured Markdown match report.

nba_api (stats.nba.com) supplies the box-score facts: arena, officials,
attendance, quarter-by-quarter score, and full per-player stats. It does not
expose head coaches or Finals MVP, so those two fields are pulled from the
small, stable "{{Infobox basketball final}}" on each "<year> NBA Finals"
Wikipedia page via the shared wikitext helpers.
"""

import re
import warnings

from nba_api.stats.endpoints import (
    boxscoresummaryv3,
    boxscoretraditionalv2,
    leaguegamefinder,
)
from nba_api.stats.static import teams as static_teams

from src.config import NBA_FINALS_YEARS
from src.wikitext import clean_wikitext, fetch_wikitext

# nba_api's pandas usage triggers noisy FutureWarnings unrelated to correctness.
warnings.filterwarnings("ignore")

TEAMS_BY_FULL_NAME = {t["full_name"]: t for t in static_teams.get_teams()}


def parse_finals_infobox(year: int) -> dict:
    """Pull champion/runner-up, coaches, series result and MVP from Wikipedia."""
    wikitext = fetch_wikitext(f"{year} NBA Finals")
    match = re.search(r"\{\{Infobox basketball final(.*?)\n\}\}", wikitext, re.S)
    if not match:
        raise ValueError("Infobox basketball final not found")
    fields: dict[str, str] = {}
    current_key = None
    for line in match.group(1).splitlines():
        field_match = re.match(r"\|\s*([\w ]+?)\s*=(.*)$", line)
        if field_match:
            current_key = field_match.group(1)
            fields[current_key] = field_match.group(2)
        elif current_key is not None:
            fields[current_key] += "\n" + line
    return {k: clean_wikitext(v) for k, v in fields.items()}


def resolve_team(full_name: str) -> dict:
    if full_name not in TEAMS_BY_FULL_NAME:
        raise ValueError(f"Unknown NBA team name from Wikipedia: {full_name!r}")
    return TEAMS_BY_FULL_NAME[full_name]


def find_clinching_game_id(season: str, team1_abbr: str, team2_abbr: str) -> str:
    """The last-played game between the two Finals teams in that season's
    playoffs is the series-clinching game."""
    finder = leaguegamefinder.LeagueGameFinder(
        season_nullable=season, season_type_nullable="Playoffs", timeout=30
    )
    df = finder.get_data_frames()[0]
    matchups = df[
        (df["TEAM_ABBREVIATION"] == team1_abbr) & (df["MATCHUP"].str.contains(team2_abbr))
    ].sort_values("GAME_DATE")
    if matchups.empty:
        raise ValueError(f"No {team1_abbr} vs {team2_abbr} playoff games found for season {season}")
    return matchups.iloc[-1]["GAME_ID"]


def fetch_box_score(game_id: str) -> dict:
    summary = boxscoresummaryv3.BoxScoreSummaryV3(game_id=game_id, timeout=30).get_dict()["boxScoreSummary"]
    traditional = boxscoretraditionalv2.BoxScoreTraditionalV2(game_id=game_id, timeout=30).get_data_frames()[0]
    return {"summary": summary, "players": traditional}


def build_facts(year: int, infobox: dict, box: dict) -> dict:
    """Flatten a Finals series and its deciding game into structured,
    arithmetic-ready facts. The Markdown report is for reading; this is for
    counting, so a total is never re-derived from prose."""
    summary = box["summary"]
    home, away = summary["homeTeam"], summary["awayTeam"]

    def team_name(team: dict) -> str:
        return f"{team['teamCity']} {team['teamName']}"

    champion = infobox.get("champion", "")
    runnerup = infobox.get("runnerup", "")
    # The infobox MVP arrives as "Stephen Curry(Golden State Warriors)",
    # "Jaylen Brown (Boston Celtics)" or a bare name -- three formats across
    # five rows, so it never equals a box-score player name and the one
    # player-to-outcome link in the data could not be joined.
    mvp = re.sub(r"\s*\(.*\)\s*$", "", infobox.get("MVP", "")).strip()

    players = []
    for _, p in box["players"].iterrows():
        if not p["MIN"] or p["MIN"] != p["MIN"]:  # skip DNPs (empty or NaN minutes)
            continue
        team = team_name(home) if p["TEAM_ID"] == home["teamId"] else team_name(away)
        players.append(
            {
                "name": p["PLAYER_NAME"],
                "team": team,
                "starter": bool(p["START_POSITION"]),
                # Pre-joined: "did this player win?" is a field lookup, not an
                # inference across a roster chunk and a result chunk.
                "team_result": "won" if team == champion else "lost",
                "is_mvp": p["PLAYER_NAME"] == mvp,
                "points": int(p["PTS"]),
                "rebounds": int(p["REB"]),
                "assists": int(p["AST"]),
            }
        )

    champion_games = infobox.get("champion_games", "")
    runnerup_games = infobox.get("runnerup_games", "")
    people = [
        {"name": p["name"], "team": p["team"], "role": "player", "team_result": p["team_result"]}
        for p in players
    ]
    for coach_field, team, result in (
        ("champion_coach", champion, "won"),
        ("runnerup_coach", runnerup, "lost"),
    ):
        if infobox.get(coach_field):
            people.append(
                {"name": infobox[coach_field], "team": team, "role": "head coach", "team_result": result}
            )

    return {
        "sport": "basketball",
        "competition": "NBA Finals",
        "year": year,
        # Shared vocabulary with the football records.
        "winner": champion,
        "loser": runnerup,
        "participants": [team_name(away), team_name(home)],
        "margin": abs(int(home["score"]) - int(away["score"])),
        "result_line": (
            f"{champion} beat {runnerup} {champion_games}-{runnerup_games} in the series"
            if champion_games
            else f"{champion} beat {runnerup}"
        ),
        "people": people,
        # Basketball-specific.
        "champion": champion,
        "runnerup": runnerup,
        "series_score": f"{champion_games}-{runnerup_games}" if champion_games else None,
        "games_played": (
            int(champion_games) + int(runnerup_games)
            if champion_games.isdigit() and runnerup_games.isdigit()
            else None
        ),
        "champion_coach": infobox.get("champion_coach", ""),
        "runnerup_coach": infobox.get("runnerup_coach", ""),
        "mvp": mvp,
        "venue": summary["arena"]["arenaName"],
        "attendance": int(summary["attendance"]) if str(summary["attendance"]).isdigit() else None,
        "officials": [o["name"] for o in summary["officials"]],
        "deciding_game": {
            "home_team": team_name(home),
            "away_team": team_name(away),
            "home_score": int(home["score"]),
            "away_score": int(away["score"]),
            "point_margin": abs(int(home["score"]) - int(away["score"])),
            # Pre-oriented, like the football scoreline.
            "score_winner_first": (
                f"{team_name(home)} {home['score']}–{away['score']} {team_name(away)}"
                if int(home["score"]) > int(away["score"])
                else f"{team_name(away)} {away['score']}–{home['score']} {team_name(home)}"
            ),
        },
        "players": players,
    }


def render_markdown(year: int, infobox: dict, box: dict) -> str:
    summary = box["summary"]
    home, away = summary["homeTeam"], summary["awayTeam"]
    arena = summary["arena"]
    officials = ", ".join(o["name"] for o in summary["officials"])

    def quarter_line(team: dict) -> str:
        periods = {p["period"]: p["score"] for p in team["periods"]}
        q_scores = "-".join(str(periods.get(q, "-")) for q in (1, 2, 3, 4))
        return f"{team['teamCity']} {team['teamName']} scored {q_scores} by quarter, final {team['score']}"

    lines = [f"# {year} NBA Finals (Basketball)", ""]
    lines += ["## Match Info"]
    lines += [
        f"- Competition: NBA Finals ({infobox.get('year', year)})",
        f"- Series dates: {infobox.get('date', '')}",
        f"- Venue: {arena['arenaName']}, {arena['arenaCity']}, {arena['arenaState']}",
        f"- Attendance: {summary['attendance']}",
        f"- Officials: {officials}",
        f"- Deciding game result: {away['teamCity']} {away['teamName']} {away['score']} – "
        f"{home['teamCity']} {home['teamName']} {home['score']}",
        f"- Quarter-by-quarter score: {quarter_line(away)}; {quarter_line(home)}",
        f"- Series result: {infobox.get('champion', '')} won {infobox.get('champion_games', '')}–"
        f"{infobox.get('runnerup_games', '')} over {infobox.get('runnerup', '')}",
        f"- Champion head coach: {infobox.get('champion_coach', '')}",
        f"- Runner-up head coach: {infobox.get('runnerup_coach', '')}",
        f"- Finals MVP: {infobox.get('MVP', '')}",
        "",
    ]

    lines += ["## Box Score"]
    players = box["players"]
    for team_id, team in ((away["teamId"], away), (home["teamId"], home)):
        lines.append(f"### {team['teamCity']} {team['teamName']}")
        team_players = players[players["TEAM_ID"] == team_id]
        for _, p in team_players.iterrows():
            if not p["MIN"] or p["MIN"] != p["MIN"]:  # skip DNPs (empty or NaN minutes)
                continue
            starter = "Starter" if p["START_POSITION"] else "Bench"
            lines.append(
                f"- {p['PLAYER_NAME']} ({starter}): {int(p['PTS'])} pts, {int(p['REB'])} reb, "
                f"{int(p['AST'])} ast, {int(p['STL'])} stl, {int(p['BLK'])} blk, {p['MIN']} min"
            )
        lines.append("")

    return "\n".join(lines)


def build_basketball_docs() -> list[dict]:
    """Fetch and render every configured NBA Finals clinching game as a Markdown report."""
    docs = []
    for year in NBA_FINALS_YEARS:
        title = f"{year} NBA Finals"
        try:
            infobox = parse_finals_infobox(year)
            champion = resolve_team(infobox["champion"])
            runnerup = resolve_team(infobox["runnerup"])
            season = f"{year - 1}-{str(year)[2:]}"
            game_id = find_clinching_game_id(season, champion["abbreviation"], runnerup["abbreviation"])
            box = fetch_box_score(game_id)
            markdown = render_markdown(year, infobox, box)
        except Exception as exc:  # noqa: BLE001 - one bad year shouldn't kill the whole ingest
            print(f"[ingest_basketball] skipping {title}: {exc}")
            continue
        docs.append(
            {
                "markdown": markdown,
                "facts": build_facts(year, infobox, box),
                "sport": "basketball",
                "competition": "NBA Finals",
                "season": season,
                "teams": f"{infobox['champion']} vs {infobox['runnerup']}",
                "source_title": title,
                "url": "https://en.wikipedia.org/wiki/" + title.replace(" ", "_"),
            }
        )
    return docs


if __name__ == "__main__":
    built_docs = build_basketball_docs()
    print(f"Built {len(built_docs)} NBA Finals match reports.")
    for d in built_docs:
        print(f"- {d['source_title']}: {d['teams']}")
