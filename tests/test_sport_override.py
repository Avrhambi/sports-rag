from app import resolve_sport
from src.plan import QueryPlan


def test_tab_is_used_when_the_question_names_no_competition():
    """Nothing to contradict, so the tab is the only signal there is."""
    assert resolve_sport("football", QueryPlan(sport="any")) == ("football", None)


def test_tab_is_used_when_it_agrees_with_the_question():
    assert resolve_sport("football", QueryPlan(sport="football")) == ("football", None)


def test_question_beats_a_stale_tab():
    """An NBA question asked with the כדורגל tab still selected used to be
    hard-filtered to football chunks and answered with a refusal."""
    assert resolve_sport("football", QueryPlan(sport="basketball")) == ("basketball", "basketball")


def test_no_tab_leaves_the_plan_in_charge():
    assert resolve_sport(None, QueryPlan(sport="basketball")) == (None, None)
