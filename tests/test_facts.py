from src.facts import basketball_block, football_block
from src.ingest_football import build_facts, full_name_index, parse_score

FOOTBALL_FINALS = [
    {
        "sport": "football", "year": 2022, "team1": "Liverpool", "team2": "Real Madrid",
        "score": "0–1", "goal_margin": 1, "winner": "Real Madrid", "loser": "Liverpool",
        "decided_on_penalties": False, "penalty_score": None,
        "attendance": 75000, "venue": "Stade de France",
    },
    {
        "sport": "football", "year": 2025, "team1": "Paris Saint-Germain", "team2": "Inter Milan",
        "score": "5–0", "goal_margin": 5, "winner": "Paris Saint-Germain", "loser": "Inter Milan",
        "decided_on_penalties": False, "penalty_score": None,
        "attendance": 64327, "venue": "Allianz Arena",
    },
    {
        "sport": "football", "year": 2026, "team1": "Paris Saint-Germain", "team2": "Arsenal",
        "score": "1–1", "goal_margin": 0, "winner": "Paris Saint-Germain", "loser": "Arsenal",
        "decided_on_penalties": True, "penalty_score": "4–3",
        "attendance": 61035, "venue": "Puskás Aréna",
    },
]

BASKETBALL_FINALS = [
    {
        "sport": "basketball", "year": 2022, "champion": "Golden State Warriors",
        "runnerup": "Boston Celtics", "series_score": "4-2", "games_played": 6,
        "winner": "Golden State Warriors", "loser": "Boston Celtics",
        "mvp": "Stephen Curry",
        "deciding_game": {
            "home_team": "Boston Celtics", "away_team": "Golden State Warriors",
            "home_score": 90, "away_score": 103, "point_margin": 13,
        },
        "players": [
            {"name": "Stephen Curry", "points": 34},
            {"name": "Jaylen Brown", "points": 34},
        ],
    },
    {
        "sport": "basketball", "year": 2024, "champion": "Boston Celtics",
        "runnerup": "Dallas Mavericks", "series_score": "4-1", "games_played": 5,
        "winner": "Boston Celtics", "loser": "Dallas Mavericks",
        "mvp": "Jaylen Brown",
        "deciding_game": {
            "home_team": "Boston Celtics", "away_team": "Dallas Mavericks",
            "home_score": 106, "away_score": 88, "point_margin": 18,
        },
        "players": [
            {"name": "Jayson Tatum", "points": 31},
            {"name": "Jaylen Brown", "points": 21},
        ],
    },
]


def test_parse_score_reads_an_en_dash_scoreline():
    assert parse_score("0–1") == (0, 1)
    assert parse_score("5-0") == (5, 0)
    assert parse_score("") is None


EMPTY_LINEUPS = {
    "team1": {"starters": [], "substitutes": [], "manager": ""},
    "team2": {"starters": [], "substitutes": [], "manager": ""},
}


def _match(**overrides) -> dict:
    match = {
        "team1": "Paris Saint-Germain", "team2": "Arsenal", "score": "1–1",
        "penalty_score": "4–3", "after_extra_time": True, "attendance": "61,035",
        "stadium": "Puskás Aréna", "referee": "Daniel Siebert",
        "goals1": [], "goals2": [], "penalties1": [], "penalties2": [],
        "lineups": EMPTY_LINEUPS,
    }
    return {**match, **overrides}


def test_build_facts_awards_a_drawn_final_to_the_shootout_winner():
    facts = build_facts(2026, _match())
    assert facts["winner"] == "Paris Saint-Germain"
    assert facts["loser"] == "Arsenal"
    assert facts["goal_margin"] == 0
    assert facts["decided_on_penalties"] is True
    assert facts["attendance"] == 61035


def test_build_facts_orients_the_scoreline_winner_first():
    """`score` is stored team1-first; an answer that names the winner first
    and then quotes it reports the match backwards."""
    facts = build_facts(2022, _match(team1="Liverpool", team2="Real Madrid", score="0–1", penalty_score=""))
    assert facts["winner"] == "Real Madrid"
    assert facts["score_winner_first"] == "Real Madrid 1–0 Liverpool"
    assert facts["result_line"] == "Real Madrid 1–0 Liverpool"


def test_build_facts_spells_out_a_shootout_in_the_oriented_score():
    assert facts_shootout_line().startswith("Paris Saint-Germain 1–1 Arsenal after extra time")
    assert "won 4–3 on penalties" in facts_shootout_line()


def facts_shootout_line() -> str:
    return build_facts(2026, _match())["score_winner_first"]


def test_build_facts_attaches_the_team_outcome_to_every_person():
    """"Did this player win?" has to be a field, not a join across chunks."""
    lineups = {
        "team1": {"starters": [{"player": "Bukayo Saka"}], "substitutes": [], "manager": "Mikel Arteta"},
        "team2": {"starters": [{"player": "Ousmane Dembélé"}], "substitutes": [], "manager": "Luis Enrique"},
    }
    people = build_facts(2026, _match(team1="Arsenal", team2="Paris Saint-Germain",
                                      penalty_score="3–4", lineups=lineups))["people"]
    by_name = {p["name"]: p for p in people}
    assert by_name["Ousmane Dembélé"]["team_result"] == "won"
    assert by_name["Bukayo Saka"]["team_result"] == "lost"
    assert by_name["Luis Enrique"]["role"] == "manager"


def test_full_name_index_expands_only_unambiguous_surnames():
    lineups = {
        "team1": {"starters": [{"player": "Nuno Mendes"}, {"player": "Gabriel Jesus"}],
                  "substitutes": [], "manager": ""},
        "team2": {"starters": [{"player": "Gabriel Magalhães"}], "substitutes": [], "manager": ""},
    }
    index = full_name_index(lineups)
    assert index["Mendes"] == "Nuno Mendes"
    # "Gabriel" is a first name shared by two players -- leave it alone rather
    # than guess, which is how a wrong player gets attached to a penalty.
    assert "Gabriel" not in index


def test_football_block_counts_titles_with_the_years_behind_them():
    block = "\n".join(football_block(FOOTBALL_FINALS))
    assert "Titles won: Paris Saint-Germain 2 (2025, 2026); Real Madrid 1 (2022)" in block


def test_football_block_attributes_losses_to_years():
    """A bare loss count made the model guess which years, and it guessed a
    year the team had actually won."""
    block = "\n".join(football_block(FOOTBALL_FINALS))
    assert "Inter Milan 1 (2025)" in block
    assert "Liverpool 1 (2022)" in block


def test_football_block_states_when_values_are_all_distinct():
    """Silence on "no maximum" read as missing data, so the model refused and
    then printed the full list it had just called unavailable."""
    block = "\n".join(football_block(FOOTBALL_FINALS))
    assert "Venues: all 3 distinct" in block


def test_football_block_picks_the_widest_margin_and_biggest_crowd():
    block = "\n".join(football_block(FOOTBALL_FINALS))
    assert "Largest goal margin: 2025, 5 goals" in block
    assert "Highest attendance: 2022 (75,000" in block


def test_football_block_lists_shootout_finals():
    assert "Decided on penalties: 2026" in "\n".join(football_block(FOOTBALL_FINALS))


def test_basketball_block_sums_points_across_separate_finals():
    """The case a language model gets wrong: 34 in 2022 plus 21 in 2024. The
    addends are spelled out so the total can't read as a single-game score."""
    block = "\n".join(basketball_block(BASKETBALL_FINALS))
    assert "Jaylen Brown: 55 total (2022: 34 + 2024: 21)" in block


def test_basketball_block_separates_single_game_highs_from_multi_year_sums():
    """A run answered "most points in a game" with a two-game sum, so the two
    live under headings that say which is which."""
    block = "\n".join(basketball_block(BASKETBALL_FINALS))
    assert "Single-game scoring (one player in one deciding game):" in block
    assert "never a single-game score" in block


def test_basketball_block_ranks_margins_and_series_length():
    block = "\n".join(basketball_block(BASKETBALL_FINALS))
    assert "Largest deciding-game margin: 2024, 18 points" in block
    assert "Longest series: 2022 (6 games, Golden State Warriors beat Boston Celtics 4-2)" in block


def test_basketball_block_pre_joins_series_length_with_the_teams():
    """The bare "2022 (4-2)" list left the champion out of the answer."""
    block = "\n".join(basketball_block(BASKETBALL_FINALS))
    assert "- 2022: 6 games — Golden State Warriors beat Boston Celtics 4-2" in block


def test_individual_record_is_keyed_by_person_not_by_year():
    """Pre-joining has to run in the direction the question asks. With only a
    per-year squad list, "did Jayson Tatum win?" was answered "no" -- the
    model scanned the MVP column rather than a comma-separated roster."""
    finals = [
        {**BASKETBALL_FINALS[0], "mvp": "Stephen Curry", "people": [
            {"name": "Stephen Curry", "team": "Golden State Warriors", "role": "player", "team_result": "won"},
            {"name": "Jayson Tatum", "team": "Boston Celtics", "role": "player", "team_result": "lost"},
        ]},
        {**BASKETBALL_FINALS[1], "mvp": "Jaylen Brown", "people": [
            {"name": "Jayson Tatum", "team": "Boston Celtics", "role": "player", "team_result": "won"},
            {"name": "Jaylen Brown", "team": "Boston Celtics", "role": "player", "team_result": "won"},
        ]},
    ]
    block = "\n".join(basketball_block(finals))
    assert "- Jayson Tatum (player): won 2024 with Boston Celtics; lost 2022 with Boston Celtics" in block
    # Both senses of "won" on one row, so answering both is a read.
    assert "- Jaylen Brown (player): won 2024 with Boston Celtics; Finals MVP 2024" in block
    assert "did not take part in any of these finals" in block


def test_basketball_block_lists_every_scorer_not_a_top_five():
    """A player outside a top-five cap read to the model as a coverage gap."""
    finals = [
        {**BASKETBALL_FINALS[0], "players": [
            {"name": f"Player {i}", "points": 30 - i} for i in range(8)
        ]},
    ]
    block = "\n".join(basketball_block(finals))
    assert "Player 7: 23 total" in block
    assert "there are no others" in block


def test_basketball_block_names_the_single_game_high():
    block = "\n".join(basketball_block(BASKETBALL_FINALS))
    assert "Most points by one player in a single deciding game: Stephen Curry, 34 (2022)" in block
