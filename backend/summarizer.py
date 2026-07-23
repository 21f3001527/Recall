"""
Map-reduce summarization.

v1 truncated the document to 6000 characters before summarizing, which
silently dropped content on anything longer than a few pages. Here we
chunk the whole document, summarize each chunk (map), then combine
those partial summaries into one structured summary (reduce) — so
quality doesn't degrade as documents get longer.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate

from config import SUMMARY_CHUNK_SIZE, SUMMARY_CHUNK_OVERLAP
from backend.models import get_llm

MAP_PROMPT = PromptTemplate(
    input_variables=["text"],
    template="""Summarize the key points of the following section of a document
in 3-5 concise bullet points. Focus on concepts, definitions, and facts —
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


def summarise_notes(full_text: str) -> str:
    llm = get_llm()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=SUMMARY_CHUNK_SIZE,
        chunk_overlap=SUMMARY_CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " "],
    )
    chunks = splitter.split_text(full_text)

    # ── Map step: summarize each chunk ──
    map_chain = MAP_PROMPT | llm
    partial_summaries = []
    for chunk in chunks:
        result = map_chain.invoke({"text": chunk})
        partial_summaries.append(result.content)

    # ── Reduce step: combine into one structured summary ──
    reduce_chain = REDUCE_PROMPT | llm
    combined = "\n\n".join(partial_summaries)
    final = reduce_chain.invoke({"summaries": combined})

    return final.content
