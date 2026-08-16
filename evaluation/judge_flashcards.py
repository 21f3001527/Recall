"""
Flashcards evaluation -- step 2: LLM-as-judge.

Scores generated flashcards on:
    - faithfulness
    - clarity
    - correctness
    - appropriate_scope

Also performs a zero-cost topic coverage check.

The judge is designed to survive Groq free-tier rate limits:
    - limited retries
    - exponential backoff
    - saves progress after every card
    - resumes previously judged cards
    - failed cards are kept with null scores
"""

import json
import logging
import time
from pathlib import Path

from pydantic import BaseModel, Field

from config import SAMPLE_DOCS_DIR
from backend.document_loader import load_pdf
from backend.models import get_llm


# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

RESULTS_DIR = Path(__file__).parent / "results"

DATASET_PATH = (
    Path(__file__).parent
    / "dataset"
    / "flashcards_eval.json"
)

RESULTS_PATH = (
    RESULTS_DIR
    / "flashcards_results.json"
)

JUDGE_RESULTS_PATH = (
    RESULTS_DIR
    / "flashcards_judge_results.json"
)

JUDGE_SUMMARY_PATH = (
    RESULTS_DIR
    / "flashcards_judge_summary.json"
)


# ---------------------------------------------------------------------
# Judge configuration
# ---------------------------------------------------------------------

# Keep this low because Groq free tier can rate-limit quickly.
MAX_RETRIES = 2

# Initial delay between retries.
INITIAL_RETRY_DELAY = 5

# Delay between successful cards.
# This reduces the chance of hitting per-minute limits.
DELAY_BETWEEN_CARDS = 2


# ---------------------------------------------------------------------
# Structured output schema
# ---------------------------------------------------------------------

class CardJudgment(BaseModel):

    faithfulness: int = Field(
        ge=1,
        le=5,
        description="1-5: is the answer supported by the source text?",
    )

    clarity: int = Field(
        ge=1,
        le=5,
        description="1-5: is the question unambiguous?",
    )

    correctness: int = Field(
        ge=1,
        le=5,
        description="1-5: is the answer accurate and complete?",
    )

    appropriate_scope: int = Field(
        ge=1,
        le=5,
        description="1-5: does it test one clear concept?",
    )

    notes: str = Field(
        description="One short sentence explaining the scores.",
    )


# ---------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------

JUDGE_PROMPT_TEMPLATE = """
You are grading a study flashcard against its source material.

SOURCE TEXT
-----------
{source}
-----------

FLASHCARD
Question: {question}
Answer: {answer}

Score the flashcard from 1 (poor) to 5 (excellent).

1. faithfulness
Is the answer directly supported by the source text?
Do not give credit for information that is not supported by the source.

2. clarity
Is the question clear, specific and unambiguous?

3. correctness
Is the answer factually correct and reasonably complete according
to the source text?

4. appropriate_scope
Does the card test one clear concept?
It should not combine several unrelated concepts and should not be
trivially obvious.

Be strict and evidence-based.
Do not automatically give 5.

Return the scores and one short explanation.
"""


# ---------------------------------------------------------------------
# Load previous judge results
# ---------------------------------------------------------------------

def load_previous_results() -> dict[int, dict]:
    """
    Load cards that were successfully judged in a previous run.

    This allows the script to resume after a rate-limit failure.
    """

    if not JUDGE_RESULTS_PATH.exists():
        return {}

    try:
        with open(
            JUDGE_RESULTS_PATH,
            "r",
            encoding="utf-8",
        ) as f:
            previous = json.load(f)

    except Exception as e:
        logger.warning(
            "Could not load previous judge results: %s",
            e,
        )
        return {}

    completed = {}

    for index, card in enumerate(previous):

        # Only resume cards that actually received scores.
        if (
            card.get("faithfulness") is not None
            and card.get("clarity") is not None
            and card.get("correctness") is not None
            and card.get("appropriate_scope") is not None
        ):
            completed[index] = card

    return completed


# ---------------------------------------------------------------------
# Save judge results
# ---------------------------------------------------------------------

def save_judge_results(judged: list[dict]) -> None:

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        JUDGE_RESULTS_PATH,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            judged,
            f,
            indent=2,
            ensure_ascii=False,
        )


# ---------------------------------------------------------------------
# Judge one card
# ---------------------------------------------------------------------

def judge_single_card(
    structured_llm,
    card: dict,
    source_text: str,
    index: int,
) -> dict | None:

    prompt = JUDGE_PROMPT_TEMPLATE.format(
        source=source_text,
        question=card["question"],
        answer=card["answer"],
    )

    for attempt in range(MAX_RETRIES + 1):

        try:

            judgment: CardJudgment = structured_llm.invoke(
                prompt
            )

            return {
                **card,
                **judgment.model_dump(),
            }

        except Exception as e:

            error_text = str(e).lower()

            # Detect rate limiting / connection problems.
            rate_limited = (
                "429" in error_text
                or "rate limit" in error_text
                or "too many requests" in error_text
                or "connection error" in error_text
            )

            if attempt >= MAX_RETRIES:

                logger.warning(
                    "Card %d failed after %d attempts: %s",
                    index + 1,
                    attempt + 1,
                    e,
                )

                return {
                    **card,
                    "faithfulness": None,
                    "clarity": None,
                    "correctness": None,
                    "appropriate_scope": None,
                    "notes": f"JUDGE_ERROR: {e}",
                }

            if rate_limited:

                delay = INITIAL_RETRY_DELAY * (
                    2 ** attempt
                )

                logger.warning(
                    "Rate limit/connection issue on card %d. "
                    "Retrying in %ss...",
                    index + 1,
                    delay,
                )

                time.sleep(delay)

            else:

                delay = 2 * (attempt + 1)

                logger.warning(
                    "Judge error on card %d. "
                    "Retrying in %ss: %s",
                    index + 1,
                    delay,
                    e,
                )

                time.sleep(delay)

    return None


# ---------------------------------------------------------------------
# Judge all cards
# ---------------------------------------------------------------------

def judge_cards(
    cards: list[dict],
    source_text: str,
) -> list[dict]:

    logger.info(
        "Creating LLM judge..."
    )

    llm = get_llm()

    structured_llm = llm.with_structured_output(
        CardJudgment
    )

    previous = load_previous_results()

    if previous:

        logger.info(
            "Found %d previously judged cards. "
            "Resuming evaluation.",
            len(previous),
        )

    judged = []

    for i, card in enumerate(cards):

        # -------------------------------------------------------------
        # Resume already completed cards
        # -------------------------------------------------------------

        if i in previous:

            judged.append(
                previous[i]
            )

            logger.info(
                "Card %d/%d already judged -- skipping.",
                i + 1,
                len(cards),
            )

            continue

        # -------------------------------------------------------------
        # Judge current card
        # -------------------------------------------------------------

        logger.info(
            "Judging card %d/%d",
            i + 1,
            len(cards),
        )

        result = judge_single_card(
            structured_llm,
            card,
            source_text,
            i,
        )

        if result is None:

            result = {
                **card,
                "faithfulness": None,
                "clarity": None,
                "correctness": None,
                "appropriate_scope": None,
                "notes": "JUDGE_ERROR",
            }

        judged.append(result)

        # -------------------------------------------------------------
        # Save immediately.
        # -------------------------------------------------------------

        save_judge_results(judged)

        logger.info(
            "Saved progress after card %d/%d.",
            i + 1,
            len(cards),
        )

        # -------------------------------------------------------------
        # Avoid hammering the free-tier API.
        # -------------------------------------------------------------

        if i < len(cards) - 1:

            time.sleep(
                DELAY_BETWEEN_CARDS
            )

    return judged


# ---------------------------------------------------------------------
# Topic coverage
# ---------------------------------------------------------------------

def check_topic_coverage(
    cards: list[dict],
    topics: list[str],
) -> dict:

    """
    Zero-cost heuristic coverage check.

    Checks whether generated flashcards touch keywords associated
    with each major topic.

    This is NOT a semantic evaluation and should not be interpreted
    as proof that a topic was correctly covered.
    """

    combined_text = " ".join(
        f"{c.get('question', '')} "
        f"{c.get('answer', '')}"
        for c in cards
    ).lower()

    covered = []
    missing = []

    for topic in topics:

        keywords = [
            word.strip(",().'\"")
            .lower()
            for word in topic.split()
            if len(word.strip(",().'\"")) > 3
        ]

        hit = any(
            keyword in combined_text
            for keyword in keywords
        )

        if hit:
            covered.append(topic)
        else:
            missing.append(topic)

    return {
        "total_topics": len(topics),
        "covered_count": len(covered),
        "coverage_pct": (
            round(
                100 * len(covered) / len(topics),
                1,
            )
            if topics
            else None
        ),
        "covered_topics": covered,
        "missing_topics": missing,
    }


# ---------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------

def summarize(
    judged: list[dict],
    coverage: dict,
) -> dict:

    metrics = [
        "faithfulness",
        "clarity",
        "correctness",
        "appropriate_scope",
    ]

    scored = [
        card
        for card in judged
        if all(
            card.get(metric) is not None
            for metric in metrics
        )
    ]

    averages = {}

    for metric in metrics:

        values = [
            card[metric]
            for card in scored
        ]

        averages[metric] = (
            round(
                sum(values) / len(values),
                2,
            )
            if values
            else None
        )

    return {
        "num_cards_total": len(judged),
        "num_cards_judged": len(scored),
        "num_cards_failed_to_judge": (
            len(judged) - len(scored)
        ),
        "average_scores": averages,
        "topic_coverage": coverage,
    }


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:

    # -----------------------------------------------------------------
    # Load generated flashcards
    # -----------------------------------------------------------------

    if not RESULTS_PATH.exists():

        raise FileNotFoundError(
            f"{RESULTS_PATH} not found. "
            "Run evaluate_flashcards.py first."
        )

    with open(
        RESULTS_PATH,
        encoding="utf-8",
    ) as f:

        results = json.load(f)

    cards = results.get(
        "cards",
        [],
    )

    if not cards:

        raise ValueError(
            "No cards found in flashcards_results.json."
        )

    # -----------------------------------------------------------------
    # Load topic dataset
    # -----------------------------------------------------------------

    if not DATASET_PATH.exists():

        raise FileNotFoundError(
            f"{DATASET_PATH} not found."
        )

    with open(
        DATASET_PATH,
        encoding="utf-8",
    ) as f:

        dataset = json.load(f)

    topics = dataset.get(
        "topics",
        [],
    )

    # -----------------------------------------------------------------
    # Load source document
    # -----------------------------------------------------------------

    doc_path = (
        SAMPLE_DOCS_DIR
        / results["doc"]
    )

    if not doc_path.exists():

        raise FileNotFoundError(
            f"Source document not found: {doc_path}"
        )

    logger.info(
        "Loading source document: %s",
        doc_path,
    )

    _, full_text, _ = load_pdf(
        str(doc_path)
    )

    # IMPORTANT:
    # Use the full source text so the judge can verify cards
    # against the complete document.
    source_text = full_text

    # -----------------------------------------------------------------
    # Judge cards
    # -----------------------------------------------------------------

    logger.info(
        "Judging %d cards...",
        len(cards),
    )

    judged = judge_cards(
        cards,
        source_text,
    )

    # -----------------------------------------------------------------
    # Topic coverage
    # -----------------------------------------------------------------

    logger.info(
        "Checking topic coverage against %d topics...",
        len(topics),
    )

    coverage = check_topic_coverage(
        cards,
        topics,
    )

    # -----------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------

    summary = summarize(
        judged,
        coverage,
    )

    # -----------------------------------------------------------------
    # Save final results
    # -----------------------------------------------------------------

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_judge_results(
        judged
    )

    with open(
        JUDGE_SUMMARY_PATH,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
            ensure_ascii=False,
        )

    # -----------------------------------------------------------------
    # Print final results
    # -----------------------------------------------------------------

    logger.info(
        "Saved per-card scores to %s",
        JUDGE_RESULTS_PATH,
    )

    logger.info(
        "Saved summary to %s",
        JUDGE_SUMMARY_PATH,
    )

    logger.info(
        "Cards successfully judged: %d/%d",
        summary["num_cards_judged"],
        summary["num_cards_total"],
    )

    logger.info(
        "Average scores: %s",
        summary["average_scores"],
    )

    logger.info(
        "Topic coverage: %s/%s (%s%%)",
        coverage["covered_count"],
        coverage["total_topics"],
        coverage["coverage_pct"],
    )

    if coverage["missing_topics"]:

        logger.warning(
            "Missing topics: %s",
            coverage["missing_topics"],
        )


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------

if __name__ == "__main__":
    main()