"""
Load a PDF, split it into chunks for retrieval, and compute a stable
content hash used as the document's ID across Chroma / SQLite.
"""

import hashlib
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHUNK_SIZE, CHUNK_OVERLAP


def hash_pdf(pdf_path: str) -> str:
    """Stable content hash — used as doc_id so re-uploading the same file
    doesn't trigger re-embedding."""
    h = hashlib.sha256()
    with open(pdf_path, "rb") as f:
        for block in iter(lambda: f.read(8192), b""):
            h.update(block)
    return h.hexdigest()[:16]


def load_pdf(pdf_path: str):
    """
    Returns (pages, full_text, doc_id)
    pages     -> list of langchain Document objects (one per page)
    full_text -> all page text joined (used by summarizer)
    doc_id    -> stable hash used to identify this document everywhere
    """
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    full_text = "\n\n".join(p.page_content for p in pages)
    doc_id = hash_pdf(pdf_path)
    return pages, full_text, doc_id


def chunk_pages(pages):
    """Split pages into retrieval-sized chunks for the vector store."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " "],
    )
    return splitter.split_documents(pages)
