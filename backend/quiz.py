"""
Quiz generator.

v1 asked the LLM to "respond with only JSON" and then manually stripped
markdown fences before calling json.loads — this silently returned an
empty quiz whenever the model added any stray text. Here we use
LangChain's structured output (backed by Groq's tool-calling support)
so the model is constrained to return a valid schema directly, with a
manual-JSON fallback for models/cases where structured output isn't
available.
"""

import json
from typing import Literal
from pydantic import BaseModel, Field

from langchain.prompts import PromptTemplate
from backend.models import get_llm


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
    """
    llm = get_llm()
    text_snippet = full_text[:6000]  # quiz only needs representative content, not the whole doc

    prompt_text = PROMPT.format(text=text_snippet, num=num_questions, difficulty=difficulty)

    try:
        structured_llm = llm.with_structured_output(QuizSet)
        result: QuizSet = structured_llm.invoke(prompt_text)
        return [q.model_dump() for q in result.questions]
    except Exception:
        # Fallback: plain-text JSON prompting for models/situations where
        # tool-calling structured output isn't available.
        return _generate_quiz_fallback(text_snippet, num_questions, difficulty, llm)


def _generate_quiz_fallback(text: str, num: int, difficulty: str, llm) -> list[dict]:
    fallback_prompt = f"""{PROMPT.format(text=text, num=num, difficulty=difficulty)}

Respond with ONLY a valid JSON array, no explanation, no markdown fences.
Each object must have exactly these keys: "question", "options", "answer".
"answer" must be just the letter A, B, C, or D."""

    raw = llm.invoke(fallback_prompt).content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return []
