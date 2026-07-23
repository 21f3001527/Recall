"""
Shared model loaders — embeddings + LLM.
Cached with st.cache_resource so they're loaded ONCE per app session,
not re-created on every Streamlit rerun (this was a performance gap in v1).
"""

import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

from config import EMBEDDING_MODEL, GROQ_MODEL, get_groq_api_key


@st.cache_resource(show_spinner=False)
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


@st.cache_resource(show_spinner=False)
def get_llm(temperature: float = 0.2):
    api_key = get_groq_api_key()
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY not found. Set it in a .env file (local) "
            "or in Streamlit secrets (cloud)."
        )
    return ChatGroq(
        model=GROQ_MODEL,
        groq_api_key=api_key,
        temperature=temperature,
    )
