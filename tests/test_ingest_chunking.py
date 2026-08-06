from src.ingest import chunk_text, clean_text


def test_chunk_text_respects_chunk_size():
    text = " ".join(f"word{i}" for i in range(1000))
    chunks = chunk_text(text, chunk_size=400, overlap=60)
    assert all(len(c.split()) <= 400 for c in chunks)
    assert chunks[0].split()[0] == "word0"


def test_chunk_text_overlaps_between_consecutive_chunks():
    text = " ".join(f"word{i}" for i in range(1000))
    chunks = chunk_text(text, chunk_size=400, overlap=60)
    first_words = chunks[0].split()
    second_words = chunks[1].split()
    assert first_words[-60:] == second_words[:60]


def test_chunk_text_empty_input_returns_no_chunks():
    assert chunk_text("") == []


def test_clean_text_strips_trailing_references_section():
    text = "Intro text.\n\n== References ==\nSome citation noise."
    assert clean_text(text) == "Intro text."


def test_clean_text_collapses_excess_blank_lines():
    text = "Para one.\n\n\n\nPara two."
    assert clean_text(text) == "Para one.\n\nPara two."
