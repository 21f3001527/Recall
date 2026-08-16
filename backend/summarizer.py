"""
Map-reduce summarization.

v1 truncated the document to 6000 characters before summarizing, which
silently dropped content on anything longer than a few pages. Here we
chunk the whole document, summarize each chunk (map), then combine
those partial summaries into one structured summary (reduce) -- so
quality doesn't degrade as documents get longer.

Reliability notes (this version)
---------------------------------
- Each map-step LLM call is retried once on failure; if a chunk still
  fails after retrying, it's skipped (logged) rather than crashing the
  whole summarization and losing every other chunk's work.
- The reduce step is also retried once before raising, since losing
  the reduce step after successfully mapping every chunk would be a
  particularly frustrating failure to hit.
- Empty/whitespace-only input is guarded against explicitly rather
  than silently producing a near-empty summary from an empty prompt.
"""

import logging

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate

from config import SUMMARY_CHUNK_SIZE, SUMMARY_CHUNK_OVERLAP
from backend.models import get_llm

logger = logging.getLogger(__name__)

MAP_PROMPT = PromptTemplate(
    input_variables=["text"],
    template="""Summarize the key points of the following section of a document
in 3-5 concise bullet points. Focus on concepts, definitions, and facts --
skip filler.

Section:
{text}

Bullet summary:""",
)

REDUCE_PROMPT = PromptTemplate(
    input_variables=["summaries"],
    template="""You are a study assistant. Below are bullet-point summaries of
consecutive sections of one document. Combine them into a single structured
summary.

Format your response EXACTLY like this:

## 📌 Summary
One paragraph overview of what this document is about.

## 🔑 Key Concepts
- Concept 1: brief explanation
- Concept 2: brief explanation
(list all important concepts, deduplicated across sections)

## 📚 Important Terms
- Term: definition
(list all technical terms with definitions)

## 💡 Key Takeaways
- Takeaway 1
- Takeaway 2
- Takeaway 3

Section summaries:
{summaries}

Structured Summary:""",
)


class SummarizationError(Exception):
    """Raised when summarization fails entirely, with a user-friendly message."""


def _invoke_with_retry(chain, inputs: dict, description: str, retries: int = 1):
    """Invoke an LLM chain, retrying once on failure. Raises on final failure."""
    last_error = None
    for attempt in range(retries + 1):
        try:
            return chain.invoke(inputs)
        except Exception as e:
            last_error = e
            logger.warning(
                "%s failed on attempt %d/%d: %s",
                description, attempt + 1, retries + 1, e,
            )
    raise last_error


def summarise_notes(full_text: str) -> str:
    if not full_text or not full_text.strip():
        raise SummarizationError(
            "There's no text to summarize -- the document appears to be empty."
        )

    llm = get_llm()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=SUMMARY_CHUNK_SIZE,
        chunk_overlap=SUMMARY_CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_text(full_text)

    if not chunks:
        raise SummarizationError(
            "The document couldn't be split into any summarizable sections."
        )

    # -- Map step: summarize each chunk --------------------------------
    map_chain = MAP_PROMPT | llm
    partial_summaries = []
    failed_chunks = 0

    for i, chunk in enumerate(chunks):
        try:
            result = _invoke_with_retry(
                map_chain, {"text": chunk}, f"Map step (chunk {i + 1}/{len(chunks)})",
            )
            partial_summaries.append(result.content)
        except Exception as e:
            failed_chunks += 1
            logger.warning(
                "Skipping chunk %d/%d after retries failed: %s",
                i + 1, len(chunks), e,
            )

    if not partial_summaries:
        raise SummarizationError(
            "Couldn't summarize any part of this document -- all sections "
            "failed. This is usually a temporary issue (e.g. rate limits) -- "
            "try again in a moment."
        )

    if failed_chunks:
        logger.warning(
            "%d of %d chunks failed to summarize and were skipped; "
            "final summary is based on the remaining %d.",
            failed_chunks, len(chunks), len(partial_summaries),
        )

    # -- Reduce step: combine into one structured summary ---------------
    reduce_chain = REDUCE_PROMPT | llm
    combined = "\n\n".join(partial_summaries)

    try:
        final = _invoke_with_retry(
            reduce_chain, {"summaries": combined}, "Reduce step",
        )
    except Exception as e:
        raise SummarizationError(
            f"Summarized all sections but failed to combine them into a "
            f"final summary. Try again. ({e})"
        ) from e

    return final.content