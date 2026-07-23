"""
Flashcards — a study mode v1 didn't have.

- Generates Q/A flashcards from the document using structured output.
- Schedules reviews using the SM-2 spaced-repetition algorithm (the same
  algorithm behind Anki), so cards you know well show up less often and
  cards you struggle with come back sooner.
"""

import logging
from datetime import datetime, timedelta, timezone
from pydantic import BaseModel, Field

from backend.models import get_llm

logger = logging.getLogger(__name__)


class Flashcard(BaseModel):
    question: str = Field(description="A short question testing one concept")
    answer: str = Field(description="A concise answer to the question")


class FlashcardSet(BaseModel):
    cards: list[Flashcard]


# ─── Sampling ──────────────────────────────────────────────────────────────

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


# ─── Generation ────────────────────────────────────────────────────────────

def generate_flashcards(
    full_text: str,
    num_cards: int = 10,
    existing_questions: list[str] | None = None,
) -> list[dict]:
    """
    Generate flashcards from the document.

    existing_questions, if provided, is used so a second/third generation
    ("Add More Flashcards") doesn't just repeat what's already there.
    """
    llm = get_llm()
    text_snippet = _sample_text(full_text)

    avoid_clause = ""
    if existing_questions:
        bullet_list = "\n".join(f"- {q}" for q in existing_questions[:40])
        avoid_clause = f"""
These flashcards already exist — do NOT repeat them or create close
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

    try:
        structured_llm = llm.with_structured_output(FlashcardSet)
        result: FlashcardSet = structured_llm.invoke(prompt)
        cards = [c.model_dump() for c in result.cards]

        # Drop anything the model returned with empty fields.
        cards = [c for c in cards if c.get("question") and c.get("answer")]
        return cards

    except Exception:
        logger.exception("Flashcard generation failed")
        return []


# ─── SM-2 spaced repetition ───────────────────────────────────────────────
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