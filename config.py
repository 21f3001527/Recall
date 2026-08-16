"""
Central configuration for Study Assistant.

All configurable paths, model names, and constants are defined here so
they can be reused across the project.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ──────────────────────────────────────────────────────────────────────────────
# Project Paths
# ──────────────────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

CHROMA_DIR = DATA_DIR / "chroma_db"
DB_PATH = DATA_DIR / "study_assistant.db"

SAMPLE_DOCS_DIR = BASE_DIR / "sample_docs"

# Ensure required directories exist
CHROMA_DIR.mkdir(exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# Models
# ──────────────────────────────────────────────────────────────────────────────

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
GROQ_MODEL = "llama-3.3-70b-versatile"
EVAL_MODEL = "llama-3.1-8b-instant"     # ← add this line

# ──────────────────────────────────────────────────────────────────────────────
# Text Chunking
# ──────────────────────────────────────────────────────────────────────────────

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# Separate chunking configuration for summarization
SUMMARY_CHUNK_SIZE = 3000
SUMMARY_CHUNK_OVERLAP = 200

# ──────────────────────────────────────────────────────────────────────────────
# Retrieval
# ──────────────────────────────────────────────────────────────────────────────

RETRIEVAL_K = 4

# ──────────────────────────────────────────────────────────────────────────────
# API Keys
# ──────────────────────────────────────────────────────────────────────────────

def get_groq_api_key() -> str:
    """
    Return the Groq API key.

    Priority:
    1. Streamlit secrets (deployment)
    2. Local .env file
    """
    try:
        import streamlit as st

        if "GROQ_API_KEY" in st.secrets:
            return st.secrets["GROQ_API_KEY"]
    except Exception:
        pass

    return os.getenv("GROQ_API_KEY", "")

