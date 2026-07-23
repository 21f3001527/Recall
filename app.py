"""
Study Assistant — Streamlit UI.

This file only handles UI + session state. All AI/data logic lives in
backend/ so the app stays easy to read and modify.
"""


import asyncio
import sys

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
import os
import tempfile
import uuid

import streamlit as st

from config import SAMPLE_DOCS_DIR
from backend import db
from backend.document_loader import load_pdf, chunk_pages
from backend.vectorstore import get_or_build_vectorstore
from backend.summarizer import summarise_notes
from backend.quiz import generate_quiz
from backend.chat import build_chat_chain, chat, reset_session
from backend.flashcards import generate_flashcards, sm2_update

db.init_db()

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────
st.set_page_config(page_title="Study Assistant", page_icon="🎓", layout="wide")

# ─── SOURCE-ON-DEMAND HELPERS ──────────────────────────────────────────────
# Sources are no longer attached to every answer automatically. They are
# only surfaced when the user's message signals they actually want them.
SOURCE_TRIGGER_WORDS = [
    "source", "sources", "citation", "citations", "cite", "cited",
    "which page", "what page", "page number", "reference", "references",
    "where did you find", "where is that from", "proof", "evidence",
    "show me the source",
]


def wants_sources(text: str) -> bool:
    """Return True if the user's message is asking for sources/citations."""
    t = text.lower()
    return any(w in t for w in SOURCE_TRIGGER_WORDS)


# ─── SESSION STATE ────────────────────────────────────────────────────────
defaults = {
    "session_id": str(uuid.uuid4()),
    "doc_name": None,
    "doc_id": None,
    "full_text": None,
    "vector_store": None,
    "n_pages": None,
    "summary": None,
    "quiz": [],
    "quiz_answers": {},
    "quiz_submitted": False,
    "chat_chain": None,
    "messages": [],
    "last_sources": [],       # sources for the most recent answer (kept in memory)
    "flashcards": [],
    "flashcard_idx": 0,
    "flashcard_show_answer": False,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


def reset_for_new_doc():
    keep = {"session_id"}
    for k, v in defaults.items():
        if k not in keep:
            st.session_state[k] = v
    reset_session(st.session_state.session_id)


# ─── SIDEBAR ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("🎓 Study Assistant")
    st.caption("Upload your notes or any PDF to get started.")
    st.divider()

    uploaded = st.file_uploader("📂 Upload PDF notes", type=["pdf"])

    if uploaded and st.session_state.doc_name != uploaded.name:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name

        with st.spinner("🔍 Indexing your notes..."):
            pages, full_text, doc_id = load_pdf(tmp_path)
            chunks = chunk_pages(pages)
            vector_store = get_or_build_vectorstore(doc_id, chunks)
        os.unlink(tmp_path)

        reset_for_new_doc()
        st.session_state.doc_name = uploaded.name
        st.session_state.doc_id = doc_id
        st.session_state.full_text = full_text
        st.session_state.vector_store = vector_store
        st.session_state.n_pages = len(pages)
        st.session_state.chat_chain = build_chat_chain(vector_store)
        st.success("✅ Ready!")
        st.rerun()

    if st.session_state.doc_name:
        st.divider()
        st.markdown(f"**📄 {st.session_state.doc_name}**")
        st.metric("Pages", st.session_state.n_pages)
        if st.button("🗑️ Clear & upload new", use_container_width=True):
            reset_for_new_doc()
            st.rerun()

    st.divider()
    st.caption("LangChain · ChromaDB · Groq · Streamlit")

# ─── MAIN ─────────────────────────────────────────────────────────────────
st.title("🎓 Study Assistant")
st.caption("Summarise notes · Generate quizzes · Flashcards · Chat with memory")
st.divider()

if st.session_state.doc_name is None:
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("### 📝 Summarise\nA structured summary with key concepts, terms, and takeaways.")
    with c2:
        st.markdown("### 🧠 Quiz\nAuto-generated MCQs to test your understanding.")
    with c3:
        st.markdown("### 🗂️ Flashcards\nSpaced-repetition flashcards that adapt to what you know.")
    with c4:
        st.markdown("### 💬 Chat\nAsk follow-up questions — it remembers the conversation.")
    st.info("👈 Upload a PDF from the sidebar to get started.")
    st.stop()

tab1, tab2, tab3, tab4 = st.tabs(["📝 Summarise", "🧠 Quiz", "🗂️ Flashcards", "💬 Chat"])

# ═══════════════════════════════════════════════════════════════════════
# TAB 1 — SUMMARISE
# ═══════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("📝 Notes Summary")

    if st.session_state.summary is None:
        st.info("Click below to generate a structured summary of your notes.")
        if st.button("✨ Generate Summary", use_container_width=True, key="gen_summary"):
            with st.spinner("Reading your notes and summarising..."):
                st.session_state.summary = summarise_notes(st.session_state.full_text)
            st.rerun()
    else:
        col1, col2 = st.columns([5, 1])
        with col2:
            if st.button("🔄 Regenerate", key="regen_summary"):
                st.session_state.summary = None
                st.rerun()
        st.markdown(st.session_state.summary)
        st.divider()
        st.download_button(
            "⬇️ Download Summary (.md)",
            data=st.session_state.summary,
            file_name="summary.md",
            mime="text/markdown",
            use_container_width=True,
        )

# ═══════════════════════════════════════════════════════════════════════
# TAB 2 — QUIZ
# ═══════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("🧠 Quiz Generator")

    if not st.session_state.quiz:
        st.info("Generate MCQ questions from your notes.")
        col_a, col_b = st.columns(2)
        with col_a:
            num_q = st.slider("Number of questions", 3, 10, 5, key="num_q")
        with col_b:
            difficulty = st.selectbox("Difficulty", ["easy", "medium", "hard"], index=1)

        if st.button("🎯 Generate Quiz", use_container_width=True, key="gen_quiz"):
            with st.spinner(f"Creating {num_q} questions..."):
                questions = generate_quiz(st.session_state.full_text, num_q, difficulty)
            if questions:
                st.session_state.quiz = questions
                st.session_state.quiz_answers = {}
                st.session_state.quiz_submitted = False
                st.rerun()
            else:
                st.error("Quiz generation failed — try again.")

    elif not st.session_state.quiz_submitted:
        questions = st.session_state.quiz
        st.info(f"Answer all {len(questions)} questions, then click Submit.")

        for i, q in enumerate(questions):
            st.markdown(f"**Q{i+1}. {q['question']}**")
            answer = st.radio(
                "Choose one:", options=q["options"], index=None,
                label_visibility="collapsed", key=f"radio_{i}",
            )
            if answer:
                st.session_state.quiz_answers[i] = answer[0]
            st.write("")

        if st.button("✅ Submit Answers", use_container_width=True, key="submit_quiz"):
            if len(st.session_state.quiz_answers) < len(questions):
                st.warning("⚠️ Please answer all questions before submitting.")
            else:
                st.session_state.quiz_submitted = True
                correct = sum(
                    1 for i, q in enumerate(questions)
                    if st.session_state.quiz_answers.get(i) == q["answer"]
                )
                db.save_quiz_result(st.session_state.doc_name, correct, len(questions))
                st.rerun()

    else:
        questions = st.session_state.quiz
        correct_count = sum(
            1 for i, q in enumerate(questions)
            if st.session_state.quiz_answers.get(i) == q["answer"]
        )
        total = len(questions)
        pct = int(correct_count / total * 100)

        if pct == 100:
            emoji, color = "🏆", "green"
        elif pct >= 60:
            emoji, color = "👍", "orange"
        else:
            emoji, color = "📖", "red"

        st.markdown(f"### {emoji} :{color}[{correct_count}/{total} correct — {pct}%]")

        for i, q in enumerate(questions):
            user_ans = st.session_state.quiz_answers.get(i, "")
            correct_ans = q["answer"]
            is_correct = user_ans == correct_ans
            icon = "✅" if is_correct else "❌"
            st.markdown(f"**{icon} Q{i+1}. {q['question']}**")
            for opt in q["options"]:
                letter = opt[0]
                if letter == correct_ans:
                    st.success(opt)
                elif letter == user_ans and not is_correct:
                    st.error(opt)
                else:
                    st.text(opt)
            st.write("")

        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Retake Same Quiz", use_container_width=True, key="retake"):
                st.session_state.quiz_answers = {}
                st.session_state.quiz_submitted = False
                st.rerun()
        with col2:
            if st.button("🆕 Generate New Quiz", use_container_width=True, key="new_quiz"):
                st.session_state.quiz = []
                st.session_state.quiz_answers = {}
                st.session_state.quiz_submitted = False
                st.rerun()

        history = db.get_quiz_history(st.session_state.doc_name)
        if len(history) > 1:
            st.divider()
            st.caption("📈 Your progress on this document")
            scores = [h["score"] / h["total"] * 100 for h in history]
            st.line_chart(scores)

# ═══════════════════════════════════════════════════════════════════════
# TAB 3 — FLASHCARDS
# ═══════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("🗂️ Flashcards")

    existing = db.get_all_flashcards(st.session_state.doc_name)

    if not existing:
        st.info("Generate flashcards from your notes. Reviews are scheduled with "
                 "spaced repetition (SM-2) — cards you know well appear less often.")
        num_cards = st.slider("Number of flashcards", 5, 20, 10, key="num_cards")
        if st.button("🗂️ Generate Flashcards", use_container_width=True, key="gen_cards"):
            with st.spinner("Creating flashcards..."):
                cards = generate_flashcards(st.session_state.full_text, num_cards)
            if cards:
                db.add_flashcards(st.session_state.doc_name, cards)
                st.rerun()
            else:
                st.error("Flashcard generation failed — try again.")
    else:
        due = db.get_due_flashcards(st.session_state.doc_name)
        st.caption(f"{len(due)} of {len(existing)} cards due for review")

        if not due:
            st.success("🎉 No cards due right now — check back later!")
        else:
            idx = st.session_state.flashcard_idx % len(due)
            card = due[idx]

            st.markdown(f"**Card {idx + 1} of {len(due)}**")
            with st.container(border=True):
                st.markdown(f"### {card['question']}")
                if st.session_state.flashcard_show_answer:
                    st.divider()
                    st.markdown(card["answer"])

            if not st.session_state.flashcard_show_answer:
                if st.button("👁️ Show Answer", use_container_width=True):
                    st.session_state.flashcard_show_answer = True
                    st.rerun()
            else:
                st.caption("How well did you know this?")
                cols = st.columns(4)
                labels = [("Again", 1), ("Hard", 3), ("Good", 4), ("Easy", 5)]
                for col, (label, quality) in zip(cols, labels):
                    with col:
                        if st.button(label, use_container_width=True, key=f"q_{quality}"):
                            ef, interval, reps, next_review = sm2_update(
                                quality, card["ease_factor"],
                                card["interval_days"], card["repetitions"],
                            )
                            db.update_flashcard_schedule(card["id"], ef, interval, reps, next_review)
                            st.session_state.flashcard_show_answer = False
                            st.rerun()

        st.divider()
        if st.button("🆕 Add More Flashcards", key="more_cards"):
            with st.spinner("Creating flashcards..."):
                cards = generate_flashcards(
                    st.session_state.full_text,
                    10,
                    existing_questions=[c["question"] for c in existing],
                )
            if cards:
                db.add_flashcards(st.session_state.doc_name, cards)
                st.rerun()
            else:
                st.error("Flashcard generation failed — try again.")
# ═══════════════════════════════════════════════════════════════════════
# TAB 4 — CHAT WITH MEMORY
# ═══════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("💬 Chat with your Notes")
    st.caption(
        "Ask anything. Follow-up questions work — the assistant remembers context. "
        "Sources aren't shown automatically — just ask e.g. *\"what's the source?\"* "
        "or *\"which page is that from?\"* whenever you want them."
    )

    if not st.session_state.messages:
        st.info("Start a conversation using the chat box at the bottom of the page.")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            # Only render the sources expander for turns where the user
            # actually asked for sources (flagged as show_sources=True).
            if msg["role"] == "assistant" and msg.get("show_sources") and msg.get("sources"):
                with st.expander("📄 Sources"):
                    for page in msg["sources"]:
                        st.caption(f"Page {page}")


# ═══════════════════════════════════════════════════════════════════════
# GLOBAL CHAT INPUT
# Must stay outside st.tabs()
# ═══════════════════════════════════════════════════════════════════════

if st.session_state.doc_name:

    question = st.chat_input("💬 Ask something about your notes...")

    if question:

        st.session_state.messages.append(
            {
                "role": "user",
                "content": question,
            }
        )

        if wants_sources(question):
            # The user is asking for sources for the last answer — don't
            # burn an LLM call, just surface what we already fetched.
            if st.session_state.last_sources:
                pages_str = ", ".join(f"page {p}" for p in st.session_state.last_sources)
                answer_text = f"Here's where my last answer came from: {pages_str}."
            else:
                answer_text = (
                    "I don't have any sources yet — ask me something about "
                    "your notes first, then I can tell you where it came from."
                )

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer_text,
                    "sources": st.session_state.last_sources,
                    "show_sources": bool(st.session_state.last_sources),
                }
            )

        else:
            with st.spinner("Thinking..."):
                result = chat(
                    st.session_state.chat_chain,
                    question,
                    st.session_state.session_id,
                )

            # Remember these sources in case the user asks for them next turn.
            st.session_state.last_sources = result["sources"]

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result["sources"],
                    "show_sources": False,  # not shown unless asked
                }
            )

        st.rerun()