"""
Persistent ChromaDB vector store.

Each uploaded document gets its own Chroma collection named after its
content hash (doc_id). If that collection already exists on disk, we
reuse it instead of re-embedding — this is the "faster / scalable"
upgrade over v1's FAISS-in-memory-only approach, where every upload
re-embedded from scratch even for a file you'd already indexed before.
"""

from langchain_chroma import Chroma

from config import CHROMA_DIR
from backend.models import get_embeddings


def collection_exists(doc_id: str) -> bool:
    client = Chroma(
        collection_name=doc_id,
        persist_directory=str(CHROMA_DIR),
        embedding_function=get_embeddings(),
    )
    return client._collection.count() > 0


def get_or_build_vectorstore(doc_id: str, chunks=None) -> Chroma:
    """
    If a collection for doc_id already exists on disk, load it (no
    re-embedding). Otherwise build it from `chunks` and persist.
    """
    embeddings = get_embeddings()

    store = Chroma(
        collection_name=doc_id,
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
    )

    if store._collection.count() == 0:
        if not chunks:
            raise ValueError(
                f"No existing collection for {doc_id} and no chunks provided to build one."
            )
        store.add_documents(chunks)

    return store


def list_indexed_docs() -> list[str]:
    """Return doc_ids of every document currently indexed on disk."""
    import chromadb
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return [c.name for c in client.list_collections()]
