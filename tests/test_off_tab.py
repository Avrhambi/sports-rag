from app import off_tab_answer, off_tab_sport
from src.plan import QueryPlan


def test_no_mismatch_when_the_question_names_no_competition():
    """Nothing contradicts the tab, so the tab is the only scope there is."""
    assert off_tab_sport("football", QueryPlan(sport="any")) is None


def test_no_mismatch_when_tab_and_question_agree():
    assert off_tab_sport("football", QueryPlan(sport="football")) is None


def test_question_about_the_other_sport_is_a_mismatch():
    """A football question on the כדורסל tab is not answerable there. It used
    to be filtered to NBA chunks and refused as if the corpus were missing
    data, which blamed the wrong thing."""
    assert off_tab_sport("basketball", QueryPlan(sport="football")) == "football"


def test_no_tab_means_no_mismatch_possible():
    assert off_tab_sport(None, QueryPlan(sport="basketball")) is None


def test_off_tab_answer_names_the_scope_the_question_and_the_way_out():
    answer = off_tab_answer("basketball", "football")
    assert "כדורסל" in answer  # the tab's scope
    assert "גמרי ה-NBA" in answer  # what that scope holds
    assert "כדורגל" in answer  # what was asked about
    assert "הכל" in answer  # the way out
