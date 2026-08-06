"""Score the RAG pipeline against a seed Hebrew Q&A set.

Two tiers:
  - Retrieval checks (always run, no API key needed): does an unfiltered
    search surface the expected sport's chunks, and do expected keywords
    show up in the retrieved context?
  - Generation checks (only run if GEMINI_API_KEY is set): Gemini judges
    each generated Hebrew answer for faithfulness and relevance against
    the retrieved context, on a 0-1 scale.

Run as: python -m eval.eval
"""

import json
import time
from pathlib import Path

from google import genai
from google.genai import errors as genai_errors

from src.config import GEMINI_API_KEY, GEMINI_MODEL_NAME
from src.generate import generate_answer
from src.retrieve import retrieve

TESTSET_PATH = Path(__file__).resolve().parent / "qa_testset.json"

# Free-tier Gemini quota is a handful of requests/minute; pace calls to stay under it
# rather than let the whole run die on the first 429.
RATE_LIMIT_DELAY_SECONDS = 13

JUDGE_PROMPT = """You are grading a Hebrew answer to a sports-rules question.
Score two things from 0 to 1 (a decimal number each), based only on the
provided context:
- faithfulness: is every claim in the answer supported by the context?
- relevance: does the answer directly address the question?

Question (Hebrew): {question}
Context: {context}
Answer (Hebrew): {answer}

Respond with exactly two lines:
faithfulness: <0-1>
relevance: <0-1>
"""


def load_testset() -> list[dict]:
    return json.loads(TESTSET_PATH.read_text(encoding="utf-8"))


def score_retrieval(item: dict, top_k: int = 4) -> dict:
    chunks = retrieve(item["question"], sport=None, top_k=top_k)
    top_sport_match = bool(chunks) and chunks[0]["sport"] == item["expected_sport"]
    context = " ".join(c["text"] for c in chunks).lower()
    keyword_hits = [kw for kw in item["expected_keywords"] if kw.lower() in context]
    return {
        "top_sport_match": top_sport_match,
        "keyword_coverage": len(keyword_hits) / len(item["expected_keywords"]),
        "chunks": chunks,
    }


def score_generation(item: dict, chunks: list[dict]) -> dict | None:
    """Ask Gemini to judge faithfulness/relevance. Returns None if no chunks, no API key, or rate-limited."""
    if not GEMINI_API_KEY or not chunks:
        return None

    try:
        time.sleep(RATE_LIMIT_DELAY_SECONDS)
        answer = generate_answer(item["question"], chunks)

        client = genai.Client(api_key=GEMINI_API_KEY)
        context = "\n".join(c["text"] for c in chunks)
        judge_prompt = JUDGE_PROMPT.format(question=item["question"], context=context, answer=answer)
        time.sleep(RATE_LIMIT_DELAY_SECONDS)
        response = client.models.generate_content(model=GEMINI_MODEL_NAME, contents=judge_prompt)
    except genai_errors.ClientError as e:
        print(f"  skipped {item['id']}: {e}")
        return None

    scores: dict = {"answer": answer}
    for line in response.text.strip().splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            try:
                scores[key.strip()] = float(value.strip())
            except ValueError:
                pass
    return scores


def main() -> None:
    testset = load_testset()
    retrieval_results = [score_retrieval(item) for item in testset]

    n = len(testset)
    sport_match_rate = sum(r["top_sport_match"] for r in retrieval_results) / n
    avg_keyword_coverage = sum(r["keyword_coverage"] for r in retrieval_results) / n

    print(f"Retrieval - top result matches expected sport: {sport_match_rate:.0%} ({n} questions)")
    print(f"Retrieval - avg expected-keyword coverage: {avg_keyword_coverage:.0%}")

    if not GEMINI_API_KEY:
        print("Generation checks skipped - set GEMINI_API_KEY in .env to enable them.")
        return

    print(f"Judging {n} generated answers with Gemini (paced for free-tier rate limits, this takes a few minutes)...")
    generation_results = []
    for item, r in zip(testset, retrieval_results):
        g = score_generation(item, r["chunks"])
        if g:
            generation_results.append(g)
    if generation_results:
        avg_faithfulness = sum(g.get("faithfulness", 0) for g in generation_results) / len(generation_results)
        avg_relevance = sum(g.get("relevance", 0) for g in generation_results) / len(generation_results)
        print(f"Generation - avg faithfulness: {avg_faithfulness:.2f} ({len(generation_results)} answers judged)")
        print(f"Generation - avg relevance: {avg_relevance:.2f}")


if __name__ == "__main__":
    main()
