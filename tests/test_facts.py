from src.facts import basketball_block, football_block
from src.ingest_football import build_facts, parse_score

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


def test_build_facts_awards_a_drawn_final_to_the_shootout_winner():
    match = {
        "team1": "Paris Saint-Germain", "team2": "Arsenal", "score": "1–1",
        "penalty_score": "4–3", "after_extra_time": True, "attendance": "61,035",
        "stadium": "Puskás Aréna", "referee": "Daniel Siebert",
    }
    facts = build_facts(2026, match)
    assert facts["winner"] == "Paris Saint-Germain"
    assert facts["loser"] == "Arsenal"
    assert facts["goal_margin"] == 0
    assert facts["decided_on_penalties"] is True
    assert facts["attendance"] == 61035


def test_football_block_counts_titles_across_finals():
    block = "\n".join(football_block(FOOTBALL_FINALS))
    assert "Titles won: Paris Saint-Germain 2, Real Madrid 1" in block


def test_football_block_picks_the_widest_margin_and_biggest_crowd():
    block = "\n".join(football_block(FOOTBALL_FINALS))
    assert "Largest goal margin: 2025 (5" in block
    assert "Highest attendance: 2022 (75,000" in block


def test_football_block_lists_shootout_finals():
    assert "Decided on penalties: 2026" in "\n".join(football_block(FOOTBALL_FINALS))


def test_basketball_block_sums_points_across_separate_finals():
    """The case a language model gets wrong: 34 in 2022 plus 21 in 2024."""
    block = "\n".join(basketball_block(BASKETBALL_FINALS))
    assert "Most points across all deciding games in range: Jaylen Brown 55" in block


def test_basketball_block_ranks_margins_and_series_length():
    block = "\n".join(basketball_block(BASKETBALL_FINALS))
    assert "Largest deciding-game margin: 2024 (18 points)" in block
    assert "Longest series: 2022 (6 games, 4-2)" in block


def test_basketball_block_names_the_single_game_high():
    block = "\n".join(basketball_block(BASKETBALL_FINALS))
    assert "Highest individual score in a deciding game: " in block
    assert "34 points (2022)" in block
