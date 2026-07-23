"""
Chat with memory using Retrieval-Augmented Generation (RAG).

Features
--------
✓ Conversation memory
✓ History-aware retrieval
✓ Similarity threshold filtering (with fallback so relevant chunks
  aren't silently dropped by an overly strict cutoff)
✓ Source page citations
✓ LangChain recommended APIs
"""

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory

from langchain.chains import (
    create_history_aware_retriever,
    create_retrieval_chain,
)

from langchain.chains.combine_documents import (
    create_stuff_documents_chain,
)

from backend.models import get_llm
from config import RETRIEVAL_K

# ---------------------------------------------------------------------
# Retrieval tuning
# ---------------------------------------------------------------------
# 0.70 cosine similarity is unrealistically strict for most embedding
# models — genuinely relevant chunks commonly score 0.3-0.55, especially
# once the history-aware retriever rewrites the question. A cutoff that
# high can silently return zero documents even for on-topic questions,
# which forces the "I couldn't find that information..." fallback every
# time. 0.35 is a much more reasonable floor; tune per your embedding
# model if needed.
SCORE_THRESHOLD = 0.35

# ---------------------------------------------------------------------
# In-memory chat history
# ---------------------------------------------------------------------

_session_store: dict[str, InMemoryChatMessageHistory] = {}


def _get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    """Return (or create) chat history for a session."""
    if session_id not in _session_store:
        _session_store[session_id] = InMemoryChatMessageHistory()

    return _session_store[session_id]


# ---------------------------------------------------------------------
# Prompt for rewriting follow-up questions
# ---------------------------------------------------------------------

CONDENSE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
Given the conversation and the latest user question,
rewrite the question so it is completely standalone.

Do NOT answer it.

Return ONLY the rewritten question.
""",
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# ---------------------------------------------------------------------
# Prompt for answering
# ---------------------------------------------------------------------

ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are an AI Study Assistant.

Answer ONLY using the provided document context.

Rules:

- Never use outside knowledge.
- If the answer is not present in the document,
  reply exactly:

I couldn't find that information in the uploaded document.

- Keep answers concise.
- Use bullet points whenever appropriate.

Context:
{context}
""",
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)


# ---------------------------------------------------------------------
# Build chain
# ---------------------------------------------------------------------

def build_chat_chain(vector_store):
    """Create the RAG pipeline."""

    llm = get_llm()

    # Primary retriever: filtered by similarity score so obviously
    # irrelevant chunks get excluded.
    strict_retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": RETRIEVAL_K,
            "score_threshold": SCORE_THRESHOLD,
        },
    )

    # Fallback retriever: plain top-k similarity, no threshold. Used only
    # when the strict retriever returns nothing, so a real question never
    # gets dropped just because every chunk scored slightly under the
    # cutoff. The LLM's ANSWER_PROMPT still decides whether the retrieved
    # context actually answers the question.
    fallback_retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": RETRIEVAL_K},
    )

    def _retrieve(query: str):
        docs = strict_retriever.invoke(query)
        if docs:
            return docs
        return fallback_retriever.invoke(query)

    retriever = RunnableLambda(_retrieve)

    history_aware_retriever = create_history_aware_retriever(
        llm,
        retriever,
        CONDENSE_PROMPT,
    )

    document_chain = create_stuff_documents_chain(
        llm,
        ANSWER_PROMPT,
    )

    rag_chain = create_retrieval_chain(
        history_aware_retriever,
        document_chain,
    )

    chain = RunnableWithMessageHistory(
        rag_chain,
        _get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    return chain


# ---------------------------------------------------------------------
# Chat API
# ---------------------------------------------------------------------

def chat(
    chain,
    question: str,
    session_id: str = "default",
):
    """
    Ask a question.

    Returns
    -------
    {
        "answer": "...",
        "sources": [1,3,5]
    }
    """

    result = chain.invoke(
        {"input": question},
        config={
            "configurable": {
                "session_id": session_id,
            }
        },
    )

    pages = []

    seen = set()

    for doc in result.get("context", []):

        page = doc.metadata.get("page")

        if page is None:
            continue

        if page not in seen:
            seen.add(page)
            pages.append(page)

    pages.sort()

    return {
        "answer": result["answer"],
        "sources": pages,
    }


# ---------------------------------------------------------------------
# Reset conversation
# ---------------------------------------------------------------------

def reset_session(session_id: str = "default"):
    """Clear chat history."""

    _session_store.pop(session_id, None)