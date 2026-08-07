from src.ingest import chunk_markdown


def test_chunk_markdown_splits_on_top_level_sections():
    doc = "# 2022 UEFA Champions League Final\n\n## Match Info\n\nShort info.\n\n## Key Events\n\nShort events."
    chunks = chunk_markdown(doc, max_words=400)
    assert chunks == [
        "2022 UEFA Champions League Final - Match Info\n\nShort info.",
        "2022 UEFA Champions League Final - Key Events\n\nShort events.",
    ]


def test_chunk_markdown_falls_back_to_subsections_when_section_too_long():
    team_a = " ".join(f"word{i}" for i in range(10))
    team_b = " ".join(f"word{i}" for i in range(10))
    doc = f"# 2022 NBA Finals\n\n## Box Score\n\n### Team A\n\n{team_a}\n\n### Team B\n\n{team_b}"
    chunks = chunk_markdown(doc, max_words=5)
    assert chunks == [
        f"2022 NBA Finals - Box Score - Team A\n\n{team_a}",
        f"2022 NBA Finals - Box Score - Team B\n\n{team_b}",
    ]


def test_chunk_markdown_no_sections_returns_no_chunks():
    assert chunk_markdown("# Just a title, no sections") == []


def test_chunk_markdown_missing_title_falls_back_to_empty_prefix():
    chunks = chunk_markdown("## Match Info\n\nSome text.", max_words=400)
    assert chunks == [" - Match Info\n\nSome text."]
