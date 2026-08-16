"""
Load a PDF, split it into chunks for retrieval, and compute a stable
content hash used as the document's ID across Chroma / SQLite.
"""

import hashlib
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHUNK_SIZE, CHUNK_OVERLAP

# Files larger than this print a warning (not a hard limit) since
# embedding on CPU scales roughly with page/chunk count and large
# files can take a long while with no progress feedback in the UI.
LARGE_FILE_WARNING_MB = 20


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

    Raises
    ------
    FileNotFoundError
        If pdf_path doesn't exist.
    ValueError
        If the PDF can't be read (corrupted, password-protected, or
        otherwise unparseable) or contains no extractable text.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    size_mb = os.path.getsize(pdf_path) / (1024 * 1024)
    if size_mb > LARGE_FILE_WARNING_MB:
        print(
            f"Warning: '{pdf_path}' is {size_mb:.1f}MB — indexing may "
            f"take a while on CPU embeddings.",
            flush=True,
        )

    try:
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
    except Exception as e:
        raise ValueError(
            f"Could not read PDF '{pdf_path}'. It may be corrupted or "
            f"password-protected. ({e})"
        ) from e

    if not pages:
        raise ValueError(f"No pages could be extracted from '{pdf_path}'.")

    full_text = "\n\n".join(p.page_content for p in pages)

    if not full_text.strip():
        raise ValueError(
            f"No extractable text found in '{pdf_path}'. It may be a "
            f"scanned/image-only PDF that needs OCR."
        )

    doc_id = hash_pdf(pdf_path)
    return pages, full_text, doc_id


def chunk_pages(pages):
    """Split pages into retrieval-sized chunks for the vector store."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(pages)