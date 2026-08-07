"""Generate a grounded Hebrew answer from retrieved English chunks via Gemini."""

from datetime import date

from google import genai

from src.config import GEMINI_API_KEY, GEMINI_MODEL_NAME

SYSTEM_PROMPT = """You are a sports history assistant covering UEFA Champions
League and NBA Finals matches. Answer strictly in Hebrew.
Use ONLY the facts in the provided source excerpts below - do not use any
outside knowledge, and do not infer facts about matches not covered by the
excerpts. If the excerpts do not contain enough information to answer the
question, say so in Hebrew instead of guessing.
Today's date is {today}. Use it to resolve relative time references in the
question (e.g. "this year", "last year", "the last 3 years") to the correct
years - do not rely on your own notion of the current date. Both
competitions' finals are played in May/June, so the season currently
underway won't have a final to report yet; if the question is about a
year or season not covered by the excerpts, say so instead of guessing."""


def build_prompt(question: str, chunks: list[dict]) -> str:
    context = "\n\n".join(f"[Source: {c['source_title']} ({c['competition']} {c['season']})]\n{c['text']}" for c in chunks)
    system_prompt = SYSTEM_PROMPT.format(today=date.today().isoformat())
    return f"{system_prompt}\n\nSource excerpts:\n{context}\n\nQuestion (Hebrew): {question}\n\nAnswer (Hebrew):"


def generate_answer(question: str, chunks: list[dict]) -> str:
    """Call Gemini to answer `question` in Hebrew, grounded only in `chunks`."""
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not set. Copy .env.example to .env and fill it in.")

    client = genai.Client(api_key=GEMINI_API_KEY)
    response = client.models.generate_content(
        model=GEMINI_MODEL_NAME,
        contents=build_prompt(question, chunks),
    )
    return response.text.strip()
