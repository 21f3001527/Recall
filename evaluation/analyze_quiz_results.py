"""
Quiz evaluation -- step 2: free sanity checks.

Runs zero-cost structural checks on the raw quiz output produced by
evaluate_quiz.py, before spending any LLM-judge API calls.

Usage:
    uv run python -m evaluation.analyze_quiz_results
"""

import json
import logging
import re
from difflib import SequenceMatcher
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_PATH = RESULTS_DIR / "quiz_results.json"

DUPLICATE_SIMILARITY_THRESHOLD = 0.85
VALID_ANSWER_LETTERS = {"A", "B", "C", "D"}
EXPECTED_NUM_OPTIONS = 4

OPTION_PREFIX_RE = re.compile(r"^([A-D])\)\s*")

# Issues containing any of these substrings are structural defects that
# should fail a CI check. Near-duplicates and generation shortfalls are
# expected/documented behavior (see README), not defects, so they're
# reported but don't fail the build.
CRITICAL_ISSUE_MARKERS = [
    "empty question text",
    "expected 4 options",
    "missing 'X) ' prefix",
    "has empty text",
    "is not one of A/B/C/D",
    "no corresponding option",
    "do not match expected",
]


def is_critical_issue(issue: str) -> bool:
    return any(marker in issue for marker in CRITICAL_ISSUE_MARKERS)


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.strip().lower(), b.strip().lower()).ratio()


def check_question(index: int, q: dict) -> list[str]:
    """Return a list of issue strings for a single question."""
    issues = []

    question_text = (q.get("question") or "").strip()
    options = q.get("options") or []
    answer = (q.get("answer") or "").strip().upper()

    if not question_text:
        issues.append("empty question text")

    if len(options) != EXPECTED_NUM_OPTIONS:
        issues.append(
            f"expected {EXPECTED_NUM_OPTIONS} options, found {len(options)}"
        )

    # Check option prefixes (A) / B) / C) / D)) and empty option text
    seen_letters = set()
    for opt in options:
        opt = (opt or "").strip()
        match = OPTION_PREFIX_RE.match(opt)
        if not match:
            issues.append(f"option missing 'X) ' prefix: {opt!r}")
            continue

        letter = match.group(1)
        seen_letters.add(letter)

        option_body = opt[match.end():].strip()
        if not option_body:
            issues.append(f"option {letter} has empty text")

    if seen_letters:
        expected_letters = set(chr(ord("A") + i) for i in range(len(options)))
        if seen_letters != expected_letters:
            issues.append(
                f"option letters {sorted(seen_letters)} do not match expected {sorted(expected_letters)}"
            )

    # Check for duplicate option text within the same question
    option_bodies = []
    for opt in options:
        match = OPTION_PREFIX_RE.match((opt or "").strip())
        body = opt[match.end():].strip() if match else (opt or "").strip()
        option_bodies.append(body)

    for i in range(len(option_bodies)):
        for j in range(i + 1, len(option_bodies)):
            if option_bodies[i] and option_bodies[j]:
                if similarity(option_bodies[i], option_bodies[j]) >= DUPLICATE_SIMILARITY_THRESHOLD:
                    issues.append(
                        f"options {i} and {j} are near-duplicates of each other"
                    )

    # Check answer validity
    if answer not in VALID_ANSWER_LETTERS:
        issues.append(f"answer {answer!r} is not one of A/B/C/D")
    elif options:
        expected_letters = set(chr(ord("A") + i) for i in range(len(options)))
        if answer not in expected_letters:
            issues.append(
                f"answer {answer!r} has no corresponding option (options cover {sorted(expected_letters)})"
            )

    # Question length sanity
    if question_text and len(question_text) < 10:
        issues.append("question text is suspiciously short (<10 chars)")

    return issues


def main() -> bool:
    """Returns True if no critical (build-breaking) issues were found,
    False otherwise. Near-duplicates and generation shortfalls are
    reported but do not fail the check -- they're expected/documented
    behavior, not structural defects.
    """
    if not RESULTS_PATH.exists():
        logger.error("Quiz results not found: %s. Run evaluate_quiz first.", RESULTS_PATH)
        return False

    try:
        with open(RESULTS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        logger.error("Failed to read %s: %s", RESULTS_PATH, e)
        return False

    questions = data.get("questions", [])
    num_requested = data.get("num_questions_requested", len(questions))
    num_generated = data.get("num_questions_generated", len(questions))
    generation_time = data.get("generation_time_seconds")

    logger.info("Loaded %d questions from %s", len(questions), RESULTS_PATH)
    logger.info(
        "Requested %d, generated %d (%.1f%%) in %ss.",
        num_requested,
        num_generated,
        100 * num_generated / num_requested if num_requested else 0.0,
        generation_time,
    )

    # ---------------------------------------------------------
    # Per-question structural checks
    # ---------------------------------------------------------
    per_question_issues: dict[int, list[str]] = {}
    for i, q in enumerate(questions):
        issues = check_question(i, q)
        if issues:
            per_question_issues[i] = issues

    # ---------------------------------------------------------
    # Cross-question duplicate detection
    # ---------------------------------------------------------
    duplicate_pairs = []
    question_texts = [(q.get("question") or "").strip() for q in questions]

    for i in range(len(question_texts)):
        for j in range(i + 1, len(question_texts)):
            if not question_texts[i] or not question_texts[j]:
                continue
            sim = similarity(question_texts[i], question_texts[j])
            if sim >= DUPLICATE_SIMILARITY_THRESHOLD:
                duplicate_pairs.append((i, j, round(sim, 3)))

    # ---------------------------------------------------------
    # Question length stats
    # ---------------------------------------------------------
    lengths = [len(q.get("question") or "") for q in questions]
    avg_len = sum(lengths) / len(lengths) if lengths else 0

    # ---------------------------------------------------------
    # Report
    # ---------------------------------------------------------
    clean_count = len(questions) - len(per_question_issues)

    logger.info("--- Free Sanity Check Summary ---")
    logger.info("Questions with no issues: %d/%d", clean_count, len(questions))
    logger.info("Average question length: %.1f characters", avg_len)
    logger.info("Duplicate question pairs (similarity >= %.2f): %d", DUPLICATE_SIMILARITY_THRESHOLD, len(duplicate_pairs))

    if per_question_issues:
        logger.warning("Issues found in %d question(s):", len(per_question_issues))
        for idx, issues in per_question_issues.items():
            q_preview = (questions[idx].get("question") or "")[:60]
            logger.warning("  [%d] %r ...", idx, q_preview)
            for issue in issues:
                logger.warning("      - %s", issue)

    if duplicate_pairs:
        logger.warning("Near-duplicate questions:")
        for i, j, sim in duplicate_pairs:
            logger.warning(
                "  [%d] <-> [%d] similarity=%.3f", i, j, sim
            )

    if num_generated < num_requested:
        logger.warning(
            "Generated fewer questions than requested (%d/%d). This is expected "
            "once the source document runs out of distinct content and is not "
            "treated as a failure.",
            num_generated,
            num_requested,
        )

    # ---------------------------------------------------------
    # Pass/fail determination
    # ---------------------------------------------------------
    critical_issue_count = sum(
        1
        for issues in per_question_issues.values()
        for issue in issues
        if is_critical_issue(issue)
    )

    if num_generated == 0:
        logger.error("No questions were generated. Failing check.")
        logger.info("Sanity check complete (FAILED). No API calls were made.")
        return False

    if critical_issue_count > 0:
        logger.error(
            "%d critical structural issue(s) found across %d question(s). Failing check.",
            critical_issue_count,
            len(per_question_issues),
        )
        logger.info("Sanity check complete (FAILED). No API calls were made.")
        return False

    logger.info("Sanity check complete (PASSED). No API calls were made.")
    return True


if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)