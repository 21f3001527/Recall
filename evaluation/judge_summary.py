"""
Summary evaluation -- step 3: LLM-as-judge.

Judges the single generated summary on four dimensions (Faithfulness,
Coverage, Conciseness, Coherence), then performs a deterministic topic
coverage check against evaluation/dataset/summary_eval.json.

Unlike quiz/flashcards, there's only one summary to judge (not many
items), so this is a single judge call with retry -- no resumable
per-item loop is needed, but every API call and file operation is still
wrapped in try/except so nothing crashes outright.

Usage:
    uv run python -m evaluation.judge_summary
    uv run python -m evaluation.judge_summary --model llama-3.1-8b-instant
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent / "results"
DATASET_DIR = Path(__file__).parent / "dataset"

SUMMARY_RESULTS_PATH = RESULTS_DIR / "summary_results.json"
JUDGE_RESULTS_PATH = RESULTS_DIR / "summary_judge_results.json"
JUDGE_SUMMARY_PATH = RESULTS_DIR / "summary_judge_summary.json"
SUMMARY_EVAL_DATASET_PATH = DATASET_DIR / "summary_eval.json"

# Use the lighter model by default for evaluation to avoid burning the
# llama-3.3-70b-versatile daily token quota (100K TPD on Groq free tier).
DEFAULT_JUDGE_MODEL = "llama-3.1-8b-instant"

MAX_RETRIES = 2
RETRY_BACKOFF_SECONDS = 5

# Bound the source excerpt to limit token usage per judge call.
SOURCE_EXCERPT_CHARS = 6000


class SummaryJudgment(BaseModel):
    faithfulness: int = Field(..., ge=1, le=5, description="Is everything in the summary actually supported by the source document (no hallucinated facts)?")
    coverage: int = Field(..., ge=1, le=5, description="Does the summary capture the source document's main topics and concepts?")
    conciseness: int = Field(..., ge=1, le=5, description="Is the summary appropriately concise, without unnecessary repetition or filler?")
    coherence: int = Field(..., ge=1, le=5, description="Is the summary well-organized, clear, and easy to follow?")
    notes: str = Field(..., description="Short explanation for the scores.")


def safe_get_judge_llm(model: str):
    try:
        from langchain_groq import ChatGroq
        llm = ChatGroq(model=model, temperature=0)
        return llm.with_structured_output(SummaryJudgment)
    except Exception as e:
        logger.error("Failed to initialize judge LLM (model=%s): %s", model, e)
        return None


def build_judge_prompt(summary: str, source_excerpt: str) -> str:
    return f"""You are grading a structured summary generated from a NumPy study document.

Source document excerpt (may be partial -- the full document is longer):
---
{source_excerpt}
---

Generated summary:
---
{summary}
---

Score the summary on four dimensions from 1 (poor) to 5 (excellent):
- faithfulness: is everything in the summary actually supported by the source (no hallucinated facts)?
- coverage: does the summary capture the source's main topics and concepts?
- conciseness: is the summary appropriately concise, without unnecessary repetition or filler?
- coherence: is the summary well-organized, clear, and easy to follow?

Provide a short explanation in notes.
"""


def safe_invoke_judge(judge, prompt: str):
    try:
        judgment = judge.invoke(prompt)
        return judgment, None
    except Exception as e:
        return None, str(e)


def load_summary_results():
    if not SUMMARY_RESULTS_PATH.exists():
        logger.error("Summary results not found: %s. Run evaluate_summary first.", SUMMARY_RESULTS_PATH)
        return None
    try:
        with open(SUMMARY_RESULTS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error("Failed to read %s: %s", SUMMARY_RESULTS_PATH, e)
        return None


def load_topic_checklist() -> list:
    if not SUMMARY_EVAL_DATASET_PATH.exists():
        logger.warning("Topic checklist not found at %s. Skipping topic coverage.", SUMMARY_EVAL_DATASET_PATH)
        return []
    try:
        with open(SUMMARY_EVAL_DATASET_PATH, "r", encoding="utf-8") as f:
            return json.load(f).get("topics", [])
    except Exception as e:
        logger.warning("Failed to read topic checklist %s: %s. Skipping topic coverage.", SUMMARY_EVAL_DATASET_PATH, e)
        return []


def save_json(path: Path, data) -> bool:
    try:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        logger.error("Failed to save %s: %s", path, e)
        return False


def check_topic_coverage(summary: str, topics: list) -> dict:
    try:
        combined_text = summary.lower()

        covered_topics = []
        missing_topics = []

        for topic in topics:
            name = topic.get("name", "unknown")
            keywords = [kw.lower() for kw in topic.get("keywords", [])]
            if any(kw in combined_text for kw in keywords):
                covered_topics.append(name)
            else:
                missing_topics.append(name)

        total = len(topics)
        covered_count = len(covered_topics)

        return {
            "total_topics": total,
            "covered_count": covered_count,
            "coverage_pct": round(100 * covered_count / total, 1) if total else 0.0,
            "covered_topics": covered_topics,
            "missing_topics": missing_topics,
        }
    except Exception as e:
        logger.warning("Topic coverage check failed: %s", e)
        return {}


def main(model: str = DEFAULT_JUDGE_MODEL) -> None:
    # ---------------------------------------------------------
    # 1. Load summary results
    # ---------------------------------------------------------
    data = load_summary_results()
    if data is None:
        return

    if data.get("status") != "success" or not data.get("summary"):
        logger.error(
            "No successful summary to judge (status=%s, error=%s).",
            data.get("status"), data.get("error"),
        )
        return

    summary = data["summary"]
    doc_name = data.get("doc")

    # ---------------------------------------------------------
    # 2. Load source document for faithfulness grounding
    # ---------------------------------------------------------
    source_excerpt = ""
    try:
        from config import SAMPLE_DOCS_DIR
        from backend.document_loader import load_pdf

        doc_path = SAMPLE_DOCS_DIR / doc_name
        _pages, full_text, _doc_id = load_pdf(str(doc_path))
        source_excerpt = full_text[:SOURCE_EXCERPT_CHARS]
    except Exception as e:
        logger.warning(
            "Could not reload source document for grounding (%s). "
            "Judging faithfulness from the summary's internal consistency only.", e,
        )

    # ---------------------------------------------------------
    # 3. Load topic checklist
    # ---------------------------------------------------------
    topics = load_topic_checklist()

    # ---------------------------------------------------------
    # 4. Judge (single call, retried)
    # ---------------------------------------------------------
    judge = safe_get_judge_llm(model)
    if judge is None:
        logger.error("Cannot proceed without a working judge LLM. Aborting.")
        return

    prompt = build_judge_prompt(summary, source_excerpt)

    judgment: Optional[SummaryJudgment] = None
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        judgment, error_message = safe_invoke_judge(judge, prompt)
        if error_message is None:
            break
        last_error = error_message
        logger.warning("Judge call failed on attempt %d/%d: %s", attempt, MAX_RETRIES, error_message)
        if attempt < MAX_RETRIES:
            try:
                time.sleep(RETRY_BACKOFF_SECONDS)
            except Exception:
                pass

    if judgment is None:
        judge_result = {"doc": doc_name, "status": "failed", "error": last_error}
        save_json(JUDGE_RESULTS_PATH, [judge_result])
        logger.error("Summary could not be judged after %d attempts: %s", MAX_RETRIES, last_error)
        return

    judge_result = {
        "doc": doc_name,
        "faithfulness": judgment.faithfulness,
        "coverage": judgment.coverage,
        "conciseness": judgment.conciseness,
        "coherence": judgment.coherence,
        "notes": judgment.notes,
        "status": "success",
    }
    save_json(JUDGE_RESULTS_PATH, [judge_result])

    # ---------------------------------------------------------
    # 5. Topic coverage + summary file
    # ---------------------------------------------------------
    coverage = check_topic_coverage(summary, topics) if topics else {}

    judge_summary = {
        "doc": doc_name,
        "model": model,
        "status": "success",
        "scores": {
            "faithfulness": judgment.faithfulness,
            "coverage": judgment.coverage,
            "conciseness": judgment.conciseness,
            "coherence": judgment.coherence,
        },
        "topic_coverage": coverage,
    }
    save_json(JUDGE_SUMMARY_PATH, judge_summary)

    # ---------------------------------------------------------
    # 6. Logging
    # ---------------------------------------------------------
    logger.info("--- Summary Judge Result ---")
    logger.info(
        "Faithfulness: %d/5, Coverage: %d/5, Conciseness: %d/5, Coherence: %d/5",
        judgment.faithfulness, judgment.coverage, judgment.conciseness, judgment.coherence,
    )
    if coverage:
        logger.info(
            "Topic coverage: %d/%d (%.1f%%)",
            coverage.get("covered_count", 0),
            coverage.get("total_topics", 0),
            coverage.get("coverage_pct", 0.0),
        )
    logger.info("Notes: %s", judgment.notes)
    logger.info("Saved judge results to %s", JUDGE_RESULTS_PATH)
    logger.info("Saved judge summary to %s", JUDGE_SUMMARY_PATH)


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(description="Judge the generated summary with an LLM.")
        parser.add_argument(
            "--model",
            default=DEFAULT_JUDGE_MODEL,
            help="Groq model to use for judging (default: llama-3.1-8b-instant to preserve daily quota).",
        )
        args = parser.parse_args()
        main(model=args.model)

    except KeyboardInterrupt:
        logger.warning("Interrupted by user.")
    except Exception as e:
        logger.exception("Unexpected error in judge_summary: %s", e)