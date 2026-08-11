from datetime import date

from src.plan import (
    QueryPlan,
    heuristic_plan,
    most_recent_completed_final_year,
    resolve_relative_years,
)


def test_sport_filter_maps_any_to_no_filter():
    assert QueryPlan(sport="any").sport_filter is None
    assert QueryPlan(sport="football").sport_filter == "football"


def test_year_strings_match_chunk_source_titles():
    assert QueryPlan(years=[2022, 2024]).year_strings == {"2022", "2024"}


def test_default_plan_is_an_unconstrained_factoid():
    plan = QueryPlan()
    assert (plan.sport_filter, plan.years, plan.intent) == (None, [], "factoid")


def test_heuristic_plan_reads_explicit_year_and_competition():
    plan = heuristic_plan("מי ניצח בגמר ליגת האלופות 2022?")
    assert plan.sport == "football"
    assert plan.years == [2022]


def test_heuristic_plan_never_guesses_a_complex_intent():
    """Surface patterns can't tell aggregation from a factoid; that is the
    planner's job, so the fallback always degrades to plain top-k."""
    assert heuristic_plan("כמה פעמים ריאל מדריד זכתה בליגת האלופות?").intent == "factoid"


def test_most_recent_completed_final_year_splits_on_july():
    assert most_recent_completed_final_year(date(2026, 6, 1)) == 2025
    assert most_recent_completed_final_year(date(2026, 8, 1)) == 2026


def test_resolve_relative_years_expands_a_hebrew_window():
    assert resolve_relative_years("בשלוש השנים האחרונות", 2025) == {"2023", "2024", "2025"}


def test_relative_years_track_whether_this_years_final_has_been_played():
    """After July this calendar year's final has happened, so "this year" is
    it and "last year" is the one before. The original mapping assumed it had
    not, and in August 2026 resolved "this year" to 2027 -- a year no corpus
    can hold -- and "last year" to 2026."""
    august = date(2026, 8, 11)
    anchor = most_recent_completed_final_year(august)
    assert resolve_relative_years("השנה", anchor, august) == {"2026"}
    assert resolve_relative_years("שנה שעברה", anchor, august) == {"2025"}


def test_relative_years_before_the_final_is_played():
    march = date(2026, 3, 1)
    anchor = most_recent_completed_final_year(march)
    # No 2026 final yet, so the most recent completed one is the best available.
    assert resolve_relative_years("השנה", anchor, march) == {"2025"}


def test_heuristic_plan_marks_itself_degraded():
    """Callers need to know the frozen fallback answered, because it reports
    `factoid` for everything and silently disables the closed-world path."""
    assert heuristic_plan("מי ניצח?").degraded is True
    assert QueryPlan().degraded is False
