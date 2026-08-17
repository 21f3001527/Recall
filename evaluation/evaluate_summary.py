"""
Summary evaluation -- step 1: generate.

Runs the actual production summarise_notes() function against the sample
document and saves the raw summary output for judge_summary.py.

Unlike quiz/flashcards, this produces a single structured summary per
document (not a list of many items), so there's no batching -- just one
call, defensively wrapped so a failure never crashes the script.

Usage:
    uv run python -m evaluation.evaluate_summary
    uv run python -m evaluation.evaluate_summary --doc "Numpy Notes.pdf"
"""

import argparse
import json
import logging
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_PATH = RESULTS_DIR / "summary_results.json"

DEFAULT_DOC_NAME = "Numpy Notes.pdf"


def safe_load_pdf(doc_path: Path):
    """Load the source PDF, returning (pages, full_text) or (None, None) on failure."""
    try:
        from backend.document_loader import load_pdf
        pages, full_text, _doc_id = load_pdf(str(doc_path))
        return pages, full_text
    except Exception as e:
        logger.error("Failed to load document %s: %s", doc_path, e)
        return None, None


def safe_summarize(full_text: str):
    """Call the production summarise_notes(), returning (summary, error_message).
    Never raises -- exceptions (including SummarizationError) are caught
    and returned as a message instead.
    """
    try:
        from backend.summarizer import summarise_notes
        summary = summarise_notes(full_text)
        return summary, None
    except Exception as e:
        return None, str(e)


def save_results(result: dict) -> bool:
    try:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(RESULTS_PATH, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        logger.error("Failed to save results to %s: %s", RESULTS_PATH, e)
        return False


def main(doc_name: str = DEFAULT_DOC_NAME) -> None:
    # ---------------------------------------------------------
    # 1. Locate + load document
    # ---------------------------------------------------------
    try:
        from config import SAMPLE_DOCS_DIR
        doc_path = SAMPLE_DOCS_DIR / doc_name
    except Exception as e:
        logger.error("Could not resolve SAMPLE_DOCS_DIR from config: %s", e)
        return

    if not doc_path.exists():
        logger.error("Sample document not found: %s", doc_path)
        return

    logger.info("Loading %s ...", doc_path)

    pages, full_text = safe_load_pdf(doc_path)
    if full_text is None:
        logger.error("Document failed to load. Aborting.")
        return

    logger.info("Loaded %d pages, %d characters.", len(pages), len(full_text))

    # ---------------------------------------------------------
    # 2. Generate summary using ACTUAL production function
    # ---------------------------------------------------------
    logger.info("Generating summary (map-reduce) ...")

    start = time.time()
    summary, error_message = safe_summarize(full_text)
    elapsed = round(time.time() - start, 2)

    if error_message is not None:
        logger.error("Summarization failed: %s", error_message)

    # ---------------------------------------------------------
    # 3. Save results (even on failure, so the attempt is recorded)
    # ---------------------------------------------------------
    result = {
        "doc": doc_name,
        "source_length_chars": len(full_text),
        "summary_length_chars": len(summary) if summary else 0,
        "generation_time_seconds": elapsed,
        "status": "success" if summary else "failed",
        "error": error_message,
        "summary": summary,
    }

    saved = save_results(result)

    # ---------------------------------------------------------
    # 4. Logging
    # ---------------------------------------------------------
    if summary:
        ratio = round(len(summary) / len(full_text), 3) if full_text else 0.0
        logger.info(
            "Generated summary: %d characters from %d source characters (ratio %.3f) in %.2fs.",
            len(summary), len(full_text), ratio, elapsed,
        )
    else:
        logger.warning("No summary was generated (%.2fs elapsed).", elapsed)

    if saved:
        logger.info("Saved summary results to %s", RESULTS_PATH)
    else:
        logger.error("Results could not be saved to disk.")


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(
            description="Generate a summary for evaluation."
        )
        parser.add_argument(
            "--doc",
            default=DEFAULT_DOC_NAME,
            help="Filename inside sample_docs/",
        )
        args = parser.parse_args()
        main(doc_name=args.doc)

    except KeyboardInterrupt:
        logger.warning("Interrupted by user.")
    except Exception as e:
        logger.exception("Unexpected error in evaluate_summary: %s", e)