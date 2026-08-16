"""
Flashcards -- a study mode v1 didn't have.

- Generates Q/A flashcards from the document using structured output.
- Schedules reviews using the SM-2 spaced-repetition algorithm (the same
  algorithm behind Anki), so cards you know well show up less often and
  cards you struggle with come back sooner.

Reliability notes (this version)
---------------------------------
- Structured output is retried once before falling back to a manual
  JSON prompt (same pattern as backend/quiz.py), instead of giving up
  after a single attempt.
- The fallback path validates each parsed card against the Flashcard
  schema, keeping whatever subset is valid rather than all-or-nothing.
- Empty/whitespace-only input is guarded against explicitly.
"""

import json
import logging
from datetime import datetime, timedelta, timezone
from pydantic import BaseModel, Field, ValidationError

from backend.models import get_llm

logger = logging.getLogger(__name__)


class Flashcard(BaseModel):
    question: str = Field(description="A short question testing one concept")
    answer: str = Field(description="A concise answer to the question")


class FlashcardSet(BaseModel):
    cards: list[Flashcard]


class FlashcardGenerationError(Exception):
    """Raised when flashcard generation fails entirely, with a friendly message."""


# --- Sampling ------------------------------------------------------------

def _sample_text(full_text: str, max_chars: int = 9000, parts: int = 3) -> str:
    """
    Sample text from across the whole document instead of just the start,
    so flashcards cover the beginning, middle, and end rather than only
    the first few pages of a long PDF.
    """
    if len(full_text) <= max_chars:
        return full_text

    chunk_size = max_chars // parts
    length = len(full_text)
    slices = []
    for i in range(parts):
        start = int(length * i / parts)
        slices.append(full_text[start:start + chunk_size])

    return "\n\n[...]\n\n".join(slices)


# --- Generation ------------------------------------------------------------

def generate_flashcards(
    full_text: str,
    num_cards: int = 10,
    existing_questions: list[str] | None = None,
) -> list[dict]:
    """
    Generate flashcards from the document.

    existing_questions, if provided, is used so a second/third generation
    ("Add More Flashcards") doesn't just repeat what's already there.

    Returns an empty list only if both the structured-output attempt
    (retried once) and the fallback attempt fail to produce any valid
    cards -- details are logged for debugging in that case.
    """
    if not full_text or not full_text.strip():
        raise FlashcardGenerationError(
            "There's no text to generate flashcards from -- the document "
            "appears to be empty."
        )

    llm = get_llm()
    text_snippet = _sample_text(full_text)

    avoid_clause = ""
    if existing_questions:
        bullet_list = "\n".join(f"- {q}" for q in existing_questions[:40])
        avoid_clause = f"""
These flashcards already exist -- do NOT repeat them or create close
rephrasings of them. Cover different concepts instead:
{bullet_list}
"""

    prompt = f"""Read the notes below and create {num_cards} flashcards for studying.

Each flashcard should test ONE concept, term, or fact with a short question
and a concise answer. Spread the questions across different sections/topics
of the notes rather than clustering on a single part.
{avoid_clause}
Notes:
{text_snippet}
"""

    # -- Attempt 1 & 2: structured output (one retry on failure) --------
    for attempt in range(2):
        try:
            structured_llm = llm.with_structured_output(FlashcardSet)
            result: FlashcardSet = structured_llm.invoke(prompt)
            cards = [c.model_dump() for c in result.cards]
            cards = [c for c in cards if c.get("question") and c.get("answer")]
            if cards:
                return cards
            logger.warning(
                "Flashcard structured output returned zero usable cards "
                "(attempt %d).", attempt + 1,
            )
        except Exception as e:
            logger.warning(
                "Flashcard structured output failed on attempt %d: %s",
                attempt + 1, e,
            )

    # -- Fallback: plain-text JSON prompting, with schema validation ----
    fallback_cards = _generate_flashcards_fallback(prompt, llm)
    if fallback_cards:
        return fallback_cards

    logger.error(
        "Flashcard generation failed entirely after structured output and "
        "fallback attempts (num_cards=%d).", num_cards,
    )
    return []


def _generate_flashcards_fallback(prompt: str, llm) -> list[dict]:
    fallback_prompt = f"""{prompt}

Respond with ONLY a valid JSON array, no explanation, no markdown fences.
Each object must have exactly these keys: "question", "answer"."""

    try:
        raw = llm.invoke(fallback_prompt).content.strip()
    except Exception as e:
        logger.warning("Flashcard fallback LLM call failed: %s", e)
        return []

    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        logger.warning("Flashcard fallback JSON parsing failed: %s | raw=%.200s", e, raw)
        return []

    if not isinstance(parsed, list):
        logger.warning("Flashcard fallback JSON was not a list: %s", type(parsed))
        return []

    valid_cards = []
    for i, item in enumerate(parsed):
        try:
            validated = Flashcard.model_validate(item)
            valid_cards.append(validated.model_dump())
        except ValidationError as e:
            logger.warning("Flashcard fallback item %d failed schema validation: %s", i, e)

    return valid_cards


# --- SM-2 spaced repetition ------------------------------------------------
# quality: how well the user recalled the card, 0-5
#   0-2 -> "again" (didn't know it)   3   -> "hard" (knew it, but barely)
#   4   -> "good"                     5   -> "easy"

def sm2_update(quality: int, ease_factor: float, interval_days: int, repetitions: int):
    """Returns (new_ease_factor, new_interval_days, new_repetitions, next_review_iso)."""
    if quality < 3:
        repetitions = 0
        interval_days = 1
    else:
        if repetitions == 0:
            interval_days = 1
        elif repetitions == 1:
            interval_days = 6
        else:
            interval_days = round(interval_days * ease_factor)
        repetitions += 1

    ease_factor = max(
        1.3,
        ease_factor + (0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02)),
    )

    next_review = (datetime.now(timezone.utc) + timedelta(days=interval_days)).isoformat()
    return ease_factor, interval_days, repetitions, next_review