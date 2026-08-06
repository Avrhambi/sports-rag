"""Generate a grounded Hebrew answer from retrieved English chunks via Gemini."""

from google import genai

from src.config import GEMINI_API_KEY, GEMINI_MODEL_NAME

SYSTEM_PROMPT = """You are a sports rules assistant. Answer strictly in Hebrew.
Use ONLY the facts in the provided source excerpts below - do not use any
outside knowledge. If the excerpts do not contain enough information to
answer the question, say so in Hebrew instead of guessing."""


def build_prompt(question: str, chunks: list[dict]) -> str:
    context = "\n\n".join(f"[Source: {c['source_title']}]\n{c['text']}" for c in chunks)
    return f"{SYSTEM_PROMPT}\n\nSource excerpts:\n{context}\n\nQuestion (Hebrew): {question}\n\nAnswer (Hebrew):"


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
