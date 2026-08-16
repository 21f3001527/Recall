"""
Lightweight SQLite persistence for quiz history and flashcards.
Keeps state across app restarts (v1 lost everything on refresh).

Documents are keyed by `doc_id` (the content hash from
document_loader.hash_pdf), matching how the vector store identifies
documents. This means quiz/flashcard history correctly follows the
same file even if it gets renamed, and correctly stays separate for
two different files that happen to share a name.

Schema note
-----------
This replaces an earlier version keyed by `doc_name` (filename). If
you have an existing data/study_assistant.db from before this change,
delete it (or the whole data/ folder) so it gets recreated with the
new schema -- CREATE TABLE IF NOT EXISTS won't migrate an existing
table's columns automatically.
"""

import sqlite3
from datetime import datetime, timezone

from config import DB_PATH


class DatabaseError(Exception):
    """Raised when a database operation fails, with a user-friendly message."""


def get_connection():
    try:
        conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn
    except sqlite3.Error as e:
        raise DatabaseError(
            f"Could not connect to the database at {DB_PATH}. ({e})"
        ) from e


def init_db():
    conn = get_connection()
    try:
        cur = conn.cursor()

        cur.execute("""
            CREATE TABLE IF NOT EXISTS quiz_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id TEXT NOT NULL,
                score INTEGER NOT NULL,
                total INTEGER NOT NULL,
                taken_at TEXT NOT NULL
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS flashcards (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id TEXT NOT NULL,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                ease_factor REAL NOT NULL DEFAULT 2.5,
                interval_days INTEGER NOT NULL DEFAULT 0,
                repetitions INTEGER NOT NULL DEFAULT 0,
                next_review TEXT NOT NULL
            )
        """)

        conn.commit()
    except sqlite3.Error as e:
        conn.rollback()
        raise DatabaseError(f"Failed to initialize database. ({e})") from e
    finally:
        conn.close()


# --- Quiz history -----------------------------------------------------

def save_quiz_result(doc_id: str, score: int, total: int):
    conn = get_connection()
    try:
        conn.execute(
            "INSERT INTO quiz_history (doc_id, score, total, taken_at) VALUES (?, ?, ?, ?)",
            (doc_id, score, total, datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()
    except sqlite3.Error as e:
        conn.rollback()
        raise DatabaseError(f"Failed to save quiz result. ({e})") from e
    finally:
        conn.close()


def get_quiz_history(doc_id: str | None = None):
    conn = get_connection()
    try:
        if doc_id:
            rows = conn.execute(
                "SELECT * FROM quiz_history WHERE doc_id = ? ORDER BY taken_at",
                (doc_id,),
            ).fetchall()
        else:
            rows = conn.execute("SELECT * FROM quiz_history ORDER BY taken_at").fetchall()
        return [dict(r) for r in rows]
    except sqlite3.Error as e:
        raise DatabaseError(f"Failed to load quiz history. ({e})") from e
    finally:
        conn.close()


# --- Flashcards ---------------------------------------------------------

def add_flashcards(doc_id: str, cards: list[dict]):
    """cards: [{"question": ..., "answer": ...}, ...]"""
    conn = get_connection()
    try:
        now = datetime.now(timezone.utc).isoformat()
        conn.executemany(
            "INSERT INTO flashcards (doc_id, question, answer, next_review) VALUES (?, ?, ?, ?)",
            [(doc_id, c["question"], c["answer"], now) for c in cards],
        )
        conn.commit()
    except sqlite3.Error as e:
        conn.rollback()
        raise DatabaseError(f"Failed to save flashcards. ({e})") from e
    finally:
        conn.close()


def get_due_flashcards(doc_id: str):
    conn = get_connection()
    try:
        now = datetime.now(timezone.utc).isoformat()
        rows = conn.execute(
            "SELECT * FROM flashcards WHERE doc_id = ? AND next_review <= ? ORDER BY next_review",
            (doc_id, now),
        ).fetchall()
        return [dict(r) for r in rows]
    except sqlite3.Error as e:
        raise DatabaseError(f"Failed to load due flashcards. ({e})") from e
    finally:
        conn.close()


def get_all_flashcards(doc_id: str):
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT * FROM flashcards WHERE doc_id = ? ORDER BY next_review", (doc_id,)
        ).fetchall()
        return [dict(r) for r in rows]
    except sqlite3.Error as e:
        raise DatabaseError(f"Failed to load flashcards. ({e})") from e
    finally:
        conn.close()


def update_flashcard_schedule(card_id: int, ease_factor: float, interval_days: int,
                               repetitions: int, next_review: str):
    conn = get_connection()
    try:
        conn.execute(
            """UPDATE flashcards
               SET ease_factor = ?, interval_days = ?, repetitions = ?, next_review = ?
               WHERE id = ?""",
            (ease_factor, interval_days, repetitions, next_review, card_id),
        )
        conn.commit()
    except sqlite3.Error as e:
        conn.rollback()
        raise DatabaseError(f"Failed to update flashcard schedule. ({e})") from e
    finally:
        conn.close()