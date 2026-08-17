"""
Quiz evaluation -- step 1: generate.

Runs the actual production generate_quiz() function against the sample
document and saves the raw quiz output for judge_quiz.py.

Every API call and file operation is wrapped in try/except so a single
failure (rate limit, network error, malformed response, disk issue)
never crashes the whole script. Progress is saved after every batch,
so a later re-run resumes instead of losing work.

Near-duplicate questions (SequenceMatcher similarity >= 0.85, same
threshold as analyze_quiz_results.py) are rejected at merge time, not
just exact text matches -- so rephrased repeats of an earlier question
don't silently pad out the results.

Usage:
    uv run python -m evaluation.evaluate_quiz
    uv run python -m evaluation.evaluate_quiz --num-questions 50 --batch-size 10
    uv run python -m evaluation.evaluate_quiz --overwrite
    uv run python -m evaluation.evaluate_quiz --wait-and-retry
"""

import argparse
import json
import logging
import re
import time
from difflib import SequenceMatcher
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_PATH = RESULTS_DIR / "quiz_results.json"

DEFAULT_DOC_NAME = "Numpy Notes.pdf"
DEFAULT_NUM_QUESTIONS = 50
DEFAULT_BATCH_SIZE = 10
DEFAULT_DIFFICULTY = "medium"

MAX_BATCH_ATTEMPTS = 2
WAIT_RETRY_BUFFER_SECONDS = 15  # extra cushion on top of Groq's suggested wait

# Same threshold used by analyze_quiz_results.py, so questions that would
# later be flagged as near-duplicates are rejected at generation time instead.
DUPLICATE_SIMILARITY_THRESHOLD = 0.85

RETRY_AFTER_RE = re.compile(r"try again in (?:(\d+)m)?([\d.]+)s")


def parse_retry_after_seconds(error_message: str):
    """Extract Groq's suggested retry wait time from a 429 error message.
    Returns None if it can't be parsed -- caller should fall back to a
    fixed default wait in that case.
    """
    try:
        match = RETRY_AFTER_RE.search(error_message)
        if not match:
            return None
        minutes = float(match.group(1)) if match.group(1) else 0.0
        seconds = float(match.group(2))
        return minutes * 60 + seconds
    except Exception as e:
        logger.warning("Could not parse retry-after time from error message: %s", e)
        return None


def safe_load_pdf(doc_path: Path):
    """Load the source PDF, returning (pages, full_text) or (None, None) on failure."""
    try:
        from backend.document_loader import load_pdf
        pages, full_text, _doc_id = load_pdf(str(doc_path))
        return pages, full_text
    except Exception as e:
        logger.error("Failed to load document %s: %s", doc_path, e)
        return None, None


def safe_generate_quiz(full_text: str, num_questions: int, difficulty: str):
    """Call the production generate_quiz(), returning a list (possibly empty)
    and never raising -- all exceptions are caught and logged, with the raw
    error message returned alongside so the caller can decide whether to
    wait-and-retry.
    """
    try:
        from backend.quiz import generate_quiz
        result = generate_quiz(full_text, num_questions=num_questions, difficulty=difficulty)
        if not isinstance(result, list):
            logger.warning("generate_quiz() returned unexpected type %s, treating as empty.", type(result))
            return [], None
        return result, None
    except Exception as e:
        return [], str(e)


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.strip().lower(), b.strip().lower()).ratio()


def is_near_duplicate(question_text: str, existing_texts: list) -> bool:
    """Check the new question against every existing one using the same
    similarity threshold as analyze_quiz_results.py, so rephrased
    duplicates are rejected here instead of surviving into the results.
    """
    for existing in existing_texts:
        if similarity(question_text, existing) >= DUPLICATE_SIMILARITY_THRESHOLD:
            return True
    return False


def load_existing_results():
    """Load previously saved quiz_results.json, or None if missing/unreadable."""
    if not RESULTS_PATH.exists():
        return None
    try:
        with open(RESULTS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning("Could not read existing %s (%s). Starting fresh.", RESULTS_PATH, e)
        return None


def save_results(result: dict) -> bool:
    """Save results to disk. Returns True on success, False on failure
    (never raises, so a save error doesn't crash mid-generation).
    """
    try:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(RESULTS_PATH, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        logger.error("Failed to save results to %s: %s", RESULTS_PATH, e)
        return False


def main(
    doc_name: str = DEFAULT_DOC_NAME,
    num_questions: int = DEFAULT_NUM_QUESTIONS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    difficulty: str = DEFAULT_DIFFICULTY,
    overwrite: bool = False,
    wait_and_retry: bool = False,
) -> None:

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
    # 2. Load existing progress (resume) or start fresh
    # ---------------------------------------------------------
    existing = None if overwrite else load_existing_results()

    if existing and existing.get("doc") == doc_name and existing.get("difficulty") == difficulty:
        questions = existing.get("questions", [])
        logger.info(
            "Found existing quiz_results.json with %d/%d questions already generated. Resuming.",
            len(questions),
            num_questions,
        )
        batch_elapsed_total = existing.get("generation_time_seconds", 0.0)
    else:
        if existing:
            logger.info("Existing quiz_results.json is for a different doc/difficulty. Starting fresh.")
        questions = []
        batch_elapsed_total = 0.0

    existing_texts = []
    for q in questions:
        try:
            text = (q.get("question") or "").strip()
            if text:
                existing_texts.append(text)
        except Exception:
            continue

    run_start_time = time.time()

    # ---------------------------------------------------------
    # 3. Generate remaining questions in batches
    # ---------------------------------------------------------
    while len(questions) < num_questions:
        remaining = num_questions - len(questions)
        this_batch_size = min(batch_size, remaining)

        logger.info(
            "Generating batch of %d questions at %s difficulty (%d/%d so far) ...",
            this_batch_size,
            difficulty,
            len(questions),
            num_questions,
        )

        batch_result = []
        stop_run = False

        for attempt in range(1, MAX_BATCH_ATTEMPTS + 1):
            batch_start = time.time()
            batch_result, error_message = safe_generate_quiz(full_text, this_batch_size, difficulty)

            if error_message is None:
                batch_elapsed_total += time.time() - batch_start
                break

            logger.warning("Batch generation failed on attempt %d: %s", attempt, error_message)

            if wait_and_retry:
                wait_seconds = parse_retry_after_seconds(error_message)
                if wait_seconds is None:
                    wait_seconds = 60.0  # fallback fixed wait if we can't parse Groq's message
                total_wait = wait_seconds + WAIT_RETRY_BUFFER_SECONDS
                logger.info("Waiting %.0f seconds before retrying ...", total_wait)
                try:
                    time.sleep(total_wait)
                except Exception as e:
                    logger.warning("Sleep interrupted: %s", e)
                continue

            # No wait-and-retry -- stop here, preserving progress so far.
            logger.error(
                "Stopping. %d/%d questions generated so far have been saved. "
                "Re-run this command later to continue, or use --wait-and-retry.",
                len(questions),
                num_questions,
            )
            stop_run = True
            break

        if stop_run:
            break

        if not batch_result:
            logger.warning("Batch produced no questions after %d attempts. Stopping.", MAX_BATCH_ATTEMPTS)
            break

        # ---------------------------------------------------------
        # 4. Merge new questions, skipping near-duplicates
        # ---------------------------------------------------------
        new_count = 0
        for q in batch_result:
            try:
                q_text = (q.get("question") or "").strip()
                if not q_text or is_near_duplicate(q_text, existing_texts):
                    continue
                questions.append(q)
                existing_texts.append(q_text)
                new_count += 1
            except Exception as e:
                logger.warning("Skipping malformed question in batch: %s", e)
                continue

        logger.info(
            "Batch added %d new question(s) (%d skipped as duplicate/invalid). Total: %d/%d.",
            new_count,
            len(batch_result) - new_count,
            len(questions),
            num_questions,
        )

        # Save progress after every batch, regardless of what happens next.
        result = {
            "doc": doc_name,
            "num_questions_requested": num_questions,
            "num_questions_generated": len(questions),
            "difficulty": difficulty,
            "generation_time_seconds": round(batch_elapsed_total, 2),
            "questions": questions,
        }
        saved = save_results(result)
        if not saved:
            logger.error("Could not save progress this batch. Continuing in-memory, will retry save next batch.")

        if new_count == 0 and not wait_and_retry:
            logger.warning("Batch produced only duplicates. Stopping to avoid an infinite loop.")
            break

    # ---------------------------------------------------------
    # 5. Final save + logging
    # ---------------------------------------------------------
    final_result = {
        "doc": doc_name,
        "num_questions_requested": num_questions,
        "num_questions_generated": len(questions),
        "difficulty": difficulty,
        "generation_time_seconds": round(batch_elapsed_total, 2),
        "questions": questions,
    }
    save_results(final_result)

    total_elapsed = round(time.time() - run_start_time, 2)

    logger.info(
        "Done. Generated %d/%d questions (this run took %.2fs).",
        len(questions),
        num_questions,
        total_elapsed,
    )
    logger.info("Saved quiz results to %s", RESULTS_PATH)

    if len(questions) < num_questions:
        logger.warning(
            "Still short by %d question(s). Re-run this command (same args) to continue.",
            num_questions - len(questions),
        )


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(
            description="Generate quiz questions for evaluation, in resumable batches."
        )

        parser.add_argument("--doc", default=DEFAULT_DOC_NAME, help="Filename inside sample_docs/")

        parser.add_argument(
            "--num-questions",
            type=int,
            default=DEFAULT_NUM_QUESTIONS,
            help="Total number of quiz questions to generate.",
        )

        parser.add_argument(
            "--batch-size",
            type=int,
            default=DEFAULT_BATCH_SIZE,
            help="Number of questions to generate per LLM call.",
        )

        parser.add_argument(
            "--difficulty",
            default=DEFAULT_DIFFICULTY,
            choices=["easy", "medium", "hard"],
            help="Quiz difficulty.",
        )

        parser.add_argument(
            "--overwrite",
            action="store_true",
            help="Ignore any existing quiz_results.json and start fresh instead of resuming.",
        )

        parser.add_argument(
            "--wait-and-retry",
            action="store_true",
            help="On a rate-limit error, automatically sleep and retry instead of stopping.",
        )

        args = parser.parse_args()

        main(
            doc_name=args.doc,
            num_questions=args.num_questions,
            batch_size=args.batch_size,
            difficulty=args.difficulty,
            overwrite=args.overwrite,
            wait_and_retry=args.wait_and_retry,
        )

    except KeyboardInterrupt:
        logger.warning("Interrupted by user. Progress up to the last saved batch is preserved.")
    except Exception as e:
        logger.exception("Unexpected error in evaluate_quiz: %s", e)