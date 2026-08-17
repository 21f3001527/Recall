"""
Summary evaluation -- step 2: free sanity checks.

Runs zero-cost structural checks on the raw summary output produced by
evaluate_summary.py, before spending any LLM-judge API calls.

The production summarise_notes() prompt (REDUCE_PROMPT) enforces a fixed
4-section markdown format, so these checks verify that structure rather
than generic text-quality heuristics.

Usage:
    uv run python -m evaluation.analyze_summary_results
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
RESULTS_PATH = RESULTS_DIR / "summary_results.json"

DUPLICATE_SIMILARITY_THRESHOLD = 0.85

EXPECTED_SECTIONS = [
    "## 📌 Summary",
    "## 🔑 Key Concepts",
    "## 📚 Important Terms",
    "## 💡 Key Takeaways",
]

# Sections that are expected to contain bullet lines (Summary is prose).
BULLET_SECTIONS = [
    "## 🔑 Key Concepts",
    "## 📚 Important Terms",
    "## 💡 Key Takeaways",
]

BULLET_LINE_RE = re.compile(r"^\s*-\s+(.*)$", re.MULTILINE)


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.strip().lower(), b.strip().lower()).ratio()


def split_into_sections(summary: str) -> dict:
    """Split the summary text into {header: body_text} using the expected
    headers as split points. Headers not found are simply absent from the
    returned dict.
    """
    sections = {}
    # Find start indices of each expected header that actually appears.
    positions = []
    for header in EXPECTED_SECTIONS:
        idx = summary.find(header)
        if idx != -1:
            positions.append((idx, header))

    positions.sort()

    for i, (start, header) in enumerate(positions):
        end = positions[i + 1][0] if i + 1 < len(positions) else len(summary)
        body = summary[start + len(header):end].strip()
        sections[header] = body

    return sections


def extract_bullets(body: str) -> list:
    return [m.strip() for m in BULLET_LINE_RE.findall(body) if m.strip()]


def main() -> None:
    if not RESULTS_PATH.exists():
        logger.error("Summary results not found: %s. Run evaluate_summary first.", RESULTS_PATH)
        return

    try:
        with open(RESULTS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        logger.error("Failed to read %s: %s", RESULTS_PATH, e)
        return

    if data.get("status") != "success" or not data.get("summary"):
        logger.error(
            "No successful summary to check (status=%s, error=%s).",
            data.get("status"), data.get("error"),
        )
        return

    summary = data["summary"]
    source_length = data.get("source_length_chars", 0)
    summary_length = data.get("summary_length_chars", len(summary))
    generation_time = data.get("generation_time_seconds")

    logger.info("Loaded summary (%d chars) from %s", summary_length, RESULTS_PATH)

    ratio = round(summary_length / source_length, 3) if source_length else 0.0
    logger.info(
        "Source: %d chars, Summary: %d chars, ratio: %.3f, generated in %ss.",
        source_length, summary_length, ratio, generation_time,
    )

    # ---------------------------------------------------------
    # 1. Section presence and order
    # ---------------------------------------------------------
    missing_sections = [h for h in EXPECTED_SECTIONS if h not in summary]
    if missing_sections:
        logger.warning("Missing expected section(s): %s", missing_sections)
    else:
        logger.info("All %d expected sections are present.", len(EXPECTED_SECTIONS))

    present_positions = [(summary.find(h), h) for h in EXPECTED_SECTIONS if h in summary]
    in_order = present_positions == sorted(present_positions)
    if not in_order:
        logger.warning("Sections are present but out of the expected order.")

    sections = split_into_sections(summary)

    # ---------------------------------------------------------
    # 2. Empty / too-short sections
    # ---------------------------------------------------------
    overview_body = sections.get("## 📌 Summary", "")
    if len(overview_body) < 20:
        logger.warning("Overview paragraph is missing or suspiciously short (<20 chars).")

    all_bullets = []
    for header in BULLET_SECTIONS:
        body = sections.get(header, "")
        bullets = extract_bullets(body)
        all_bullets.extend(bullets)
        if header in sections and not bullets:
            logger.warning("Section %r is present but has no bullet points.", header)
        elif header in sections:
            logger.info("Section %r has %d bullet point(s).", header, len(bullets))

    # ---------------------------------------------------------
    # 3. Near-duplicate bullets (across Key Concepts / Terms / Takeaways)
    # ---------------------------------------------------------
    duplicate_pairs = []
    for i in range(len(all_bullets)):
        for j in range(i + 1, len(all_bullets)):
            sim = similarity(all_bullets[i], all_bullets[j])
            if sim >= DUPLICATE_SIMILARITY_THRESHOLD:
                duplicate_pairs.append((i, j, round(sim, 3)))

    if duplicate_pairs:
        logger.warning(
            "Found %d near-duplicate bullet pair(s) (similarity >= %.2f):",
            len(duplicate_pairs), DUPLICATE_SIMILARITY_THRESHOLD,
        )
        for i, j, sim in duplicate_pairs:
            logger.warning("  %r <-> %r (%.3f)", all_bullets[i][:60], all_bullets[j][:60], sim)
    else:
        logger.info("No near-duplicate bullets found across Key Concepts / Terms / Takeaways.")

    # ---------------------------------------------------------
    # 4. Length ratio sanity
    # ---------------------------------------------------------
    if ratio > 0.5:
        logger.warning(
            "Summary is %.0f%% the length of the source -- unusually long for a summary.", ratio * 100
        )
    elif source_length and summary_length < 100:
        logger.warning("Summary is suspiciously short (<100 chars) relative to source.")

    # ---------------------------------------------------------
    # Report
    # ---------------------------------------------------------
    logger.info("--- Free Sanity Check Summary ---")
    logger.info("Sections present: %d/%d", len(EXPECTED_SECTIONS) - len(missing_sections), len(EXPECTED_SECTIONS))
    logger.info("Total bullets across Key Concepts/Terms/Takeaways: %d", len(all_bullets))
    logger.info("Near-duplicate bullet pairs: %d", len(duplicate_pairs))
    logger.info(
        "Note: per-chunk map-step failures (if any) are only visible in the "
        "evaluate_summary run's console log, not in the saved JSON -- check "
        "that run's output if you suspect content was silently dropped."
    )
    logger.info("Sanity check complete. No API calls were made.")


if __name__ == "__main__":
    main()