"""
Flashcards evaluation -- step 1: generate.

Runs the actual production `generate_flashcards()` function against the
sample document and saves the raw output. This mirrors evaluate_chat.py:
it captures what the real pipeline produces, so judge_flashcards.py can
score it afterwards without needing to re-run generation.

Usage:
    uv run python -m evaluation.evaluate_flashcards
    uv run python -m evaluation.evaluate_flashcards --num-cards 20
"""

import argparse
import json
import logging
import time
from pathlib import Path

from config import SAMPLE_DOCS_DIR
from backend.document_loader import load_pdf
from backend.flashcards import generate_flashcards, FlashcardGenerationError

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_PATH = RESULTS_DIR / "flashcards_results.json"

DEFAULT_DOC_NAME = "Numpy Notes.pdf"
DEFAULT_NUM_CARDS = 15


def main(doc_name: str = DEFAULT_DOC_NAME, num_cards: int = DEFAULT_NUM_CARDS) -> None:
    doc_path = SAMPLE_DOCS_DIR / doc_name
    if not doc_path.exists():
        raise FileNotFoundError(f"Sample doc not found: {doc_path}")

    logger.info("Loading %s ...", doc_path)
    pages, full_text, _ = load_pdf(str(doc_path))
    logger.info("Loaded %d pages, %d characters.", len(pages), len(full_text))

    logger.info("Generating %d flashcards ...", num_cards)
    start = time.time()
    try:
        cards = generate_flashcards(full_text, num_cards=num_cards)
    except FlashcardGenerationError as e:
        logger.error("Flashcard generation raised: %s", e)
        cards = []
    elapsed = round(time.time() - start, 2)

    result = {
        "doc": doc_name,
        "num_cards_requested": num_cards,
        "num_cards_generated": len(cards),
        "generation_time_seconds": elapsed,
        "cards": cards,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    logger.info(
        "Generated %d/%d cards in %.2fs. Saved to %s",
        len(cards), num_cards, elapsed, RESULTS_PATH,
    )
    if len(cards) < num_cards:
        logger.warning(
            "Fewer cards than requested -- check logs above for generation "
            "failures before proceeding to judge_flashcards.py."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate flashcards for evaluation.")
    parser.add_argument("--doc", default=DEFAULT_DOC_NAME, help="Filename inside sample_docs/")
    parser.add_argument("--num-cards", type=int, default=DEFAULT_NUM_CARDS)
    args = parser.parse_args()
    main(doc_name=args.doc, num_cards=args.num_cards)