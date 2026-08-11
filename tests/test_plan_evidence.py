from src.plan import QueryPlan
from src.retrieve import plan_evidence_ids

CHUNKS = [
    {"id": "f-2022-info", "sport": "football", "source_title": "2022 UEFA Champions League final"},
    {"id": "f-2022-lineups", "sport": "football", "source_title": "2022 UEFA Champions League final"},
    {"id": "f-2024-info", "sport": "football", "source_title": "2024 UEFA Champions League final"},
    {"id": "b-2022-info", "sport": "basketball", "source_title": "2022 NBA Finals"},
    {"id": "b-2024-info", "sport": "basketball", "source_title": "2024 NBA Finals"},
]


def ids(plan: QueryPlan, sport: str | None = None) -> set[str]:
    return plan_evidence_ids(CHUNKS, plan, sport)


def test_planned_year_pulls_every_section_of_that_report():
    """Not just the best-matching section -- a result lives in Match Info
    while the roster lives in Lineups."""
    assert ids(QueryPlan(sport="football", years=[2022]), "football") == {"f-2022-info", "f-2022-lineups"}


def test_planned_years_span_several_reports():
    assert ids(QueryPlan(sport="football", years=[2022, 2024]), "football") == {
        "f-2022-info",
        "f-2022-lineups",
        "f-2024-info",
    }


def test_a_yearless_factoid_still_sees_every_final():
    """"מי ניצח בגמר" points at no single final, so there is nothing for
    similarity to find. Left to plain top-k it returned an arbitrary four
    chunks, which the answerer reported as though they were the whole story."""
    assert ids(QueryPlan(sport="football", intent="factoid"), "football") == {
        "f-2022-info",
        "f-2022-lineups",
        "f-2024-info",
    }


def test_a_yearless_sportless_factoid_reaches_both_sports():
    assert ids(QueryPlan(sport="any", intent="factoid")) == {
        "f-2022-info",
        "f-2022-lineups",
        "f-2024-info",
        "b-2022-info",
        "b-2024-info",
    }


def test_yearless_aggregate_still_sees_every_final():
    """"Which final had the biggest crowd" names no year but needs them all."""
    assert ids(QueryPlan(sport="football", intent="aggregate"), "football") == {
        "f-2022-info",
        "f-2022-lineups",
        "f-2024-info",
    }


def test_cross_sport_plan_keeps_both_sports():
    assert ids(QueryPlan(sport="any", years=[2024], intent="comparison")) == {"f-2024-info", "b-2024-info"}


def test_explicit_sport_filter_narrows_a_cross_sport_plan():
    """The UI's sport tab overrides the plan, so evidence follows the tab."""
    assert ids(QueryPlan(sport="any", years=[2024], intent="comparison"), "basketball") == {"b-2024-info"}
