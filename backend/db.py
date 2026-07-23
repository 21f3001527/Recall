"""
Lightweight SQLite persistence for quiz history and flashcards.
Keeps state across app restarts (v1 lost everything on refresh).
"""

import sqlite3
from datetime import datetime, timezone

from config import DB_PATH


def get_connection():
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS quiz_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_name TEXT NOT NULL,
            score INTEGER NOT NULL,
            total INTEGER NOT NULL,
            taken_at TEXT NOT NULL
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS flashcards (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_name TEXT NOT NULL,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            ease_factor REAL NOT NULL DEFAULT 2.5,
            interval_days INTEGER NOT NULL DEFAULT 0,
            repetitions INTEGER NOT NULL DEFAULT 0,
            next_review TEXT NOT NULL
        )
    """)

    conn.commit()
    conn.close()


# ─── Quiz history ─────────────────────────────────────────────────────────

def save_quiz_result(doc_name: str, score: int, total: int):
    conn = get_connection()
    conn.execute(
        "INSERT INTO quiz_history (doc_name, score, total, taken_at) VALUES (?, ?, ?, ?)",
        (doc_name, score, total, datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()
    conn.close()


def get_quiz_history(doc_name: str | None = None):
    conn = get_connection()
    if doc_name:
        rows = conn.execute(
            "SELECT * FROM quiz_history WHERE doc_name = ? ORDER BY taken_at",
            (doc_name,),
        ).fetchall()
    else:
        rows = conn.execute("SELECT * FROM quiz_history ORDER BY taken_at").fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ─── Flashcards ───────────────────────────────────────────────────────────

def add_flashcards(doc_name: str, cards: list[dict]):
    """cards: [{"question": ..., "answer": ...}, ...]"""
    conn = get_connection()
    now = datetime.now(timezone.utc).isoformat()
    conn.executemany(
        "INSERT INTO flashcards (doc_name, question, answer, next_review) VALUES (?, ?, ?, ?)",
        [(doc_name, c["question"], c["answer"], now) for c in cards],
    )
    conn.commit()
    conn.close()


def get_due_flashcards(doc_name: str):
    conn = get_connection()
    now = datetime.now(timezone.utc).isoformat()
    rows = conn.execute(
        "SELECT * FROM flashcards WHERE doc_name = ? AND next_review <= ? ORDER BY next_review",
        (doc_name, now),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_all_flashcards(doc_name: str):
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM flashcards WHERE doc_name = ? ORDER BY next_review", (doc_name,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def update_flashcard_schedule(card_id: int, ease_factor: float, interval_days: int,
                               repetitions: int, next_review: str):
    conn = get_connection()
    conn.execute(
        """UPDATE flashcards
           SET ease_factor = ?, interval_days = ?, repetitions = ?, next_review = ?
           WHERE id = ?""",
        (ease_factor, interval_days, repetitions, next_review, card_id),
    )
    conn.commit()
    conn.close()
