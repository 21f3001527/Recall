"""
Quiz generator.

v1 asked the LLM to "respond with only JSON" and then manually stripped
markdown fences before calling json.loads -- this silently returned an
empty quiz whenever the model added any stray text. Here we use
LangChain's structured output (backed by Groq's tool-calling support)
so the model is constrained to return a valid schema directly, with a
manual-JSON fallback for models/cases where structured output isn't
available.

Reliability notes (this version)
---------------------------------
- Failures are logged (not silently swallowed) so "why did quiz
  generation fail" is debuggable instead of a mystery.
- One retry is attempted before giving up entirely -- LLM output is
  naturally variable, and a single retry meaningfully cuts down on
  transient malformed-output failures.
- The fallback path validates each parsed question against the same
  QuizQuestion schema used by structured output, and returns whatever
  subset of questions actually validated instead of an all-or-nothing
  json.loads() success/failure.
"""

import json
import logging
from typing import Literal
from pydantic import BaseModel, Field, ValidationError

from langchain.prompts import PromptTemplate
from backend.models import get_llm

logger = logging.getLogger(__name__)


class QuizQuestion(BaseModel):
    question: str = Field(description="The quiz question text")
    options: list[str] = Field(
        description="Exactly 4 options, formatted as 'A) ...', 'B) ...', 'C) ...', 'D) ...'"
    )
    answer: Literal["A", "B", "C", "D"] = Field(
        description="The correct option letter"
    )


class QuizSet(BaseModel):
    questions: list[QuizQuestion]


PROMPT = PromptTemplate(
    input_variables=["text", "num", "difficulty"],
    template="""You are a quiz generator. Read the notes below and create {num}
multiple choice questions at {difficulty} difficulty.

Each question must have exactly 4 options labeled A) B) C) D), and one
correct answer.

Notes:
{text}
""",
)


def generate_quiz(full_text: str, num_questions: int = 5, difficulty: str = "medium") -> list[dict]:
    """
    Returns a list of dicts: [{"question": ..., "options": [...], "answer": "A"}, ...]

    Returns an empty list only if both the structured-output attempt and
    the fallback attempt (each retried once) fail to produce any valid
    questions -- in which case details are logged for debugging.
    """
    llm = get_llm()
    text_snippet = full_text[:6000]  # quiz only needs representative content, not the whole doc

    prompt_text = PROMPT.format(text=text_snippet, num=num_questions, difficulty=difficulty)

    # --- Attempt 1 & 2: structured output (one retry on failure) -----
    for attempt in range(2):
        try:
            structured_llm = llm.with_structured_output(QuizSet)
            result: QuizSet = structured_llm.invoke(prompt_text)
            if result.questions:
                return [q.model_dump() for q in result.questions]
            logger.warning(
                "Quiz structured output returned zero questions (attempt %d).",
                attempt + 1,
            )
        except Exception as e:
            logger.warning(
                "Quiz structured output failed on attempt %d: %s",
                attempt + 1,
                e,
            )

    # --- Fallback: plain-text JSON prompting, with schema validation --
    for attempt in range(2):
        questions = _generate_quiz_fallback(text_snippet, num_questions, difficulty, llm)
        if questions:
            return questions
        logger.warning("Quiz fallback attempt %d produced no valid questions.", attempt + 1)

    logger.error(
        "Quiz generation failed entirely after structured output and fallback "
        "attempts (num_questions=%d, difficulty=%s).",
        num_questions,
        difficulty,
    )
    return []


def _generate_quiz_fallback(text: str, num: int, difficulty: str, llm) -> list[dict]:
    """
    Plain-text JSON prompting fallback. Validates each parsed item against
    QuizQuestion and returns only the ones that pass -- a few malformed
    questions in the response no longer discards the whole batch.
    """
    fallback_prompt = f"""{PROMPT.format(text=text, num=num, difficulty=difficulty)}

Respond with ONLY a valid JSON array, no explanation, no markdown fences.
Each object must have exactly these keys: "question", "options", "answer".
"answer" must be just the letter A, B, C, or D."""

    try:
        raw = llm.invoke(fallback_prompt).content.strip()
    except Exception as e:
        logger.warning("Quiz fallback LLM call failed: %s", e)
        return []

    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        logger.warning("Quiz fallback JSON parsing failed: %s | raw=%.200s", e, raw)
        return []

    if not isinstance(parsed, list):
        logger.warning("Quiz fallback JSON was not a list: %s", type(parsed))
        return []

    valid_questions = []
    for i, item in enumerate(parsed):
        try:
            validated = QuizQuestion.model_validate(item)
            valid_questions.append(validated.model_dump())
        except ValidationError as e:
            logger.warning("Quiz fallback item %d failed schema validation: %s", i, e)

    return valid_questions