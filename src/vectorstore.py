"""
Stage 5 — Store everything in ChromaDB.

Persisted to disk (CHROMA_PERSIST_DIR), so it survives a restart —
that's the whole point of the "restart test" in the guide.

Duplicate-ingestion guard: every chunk gets a stable ID derived from its
file name, page number, and position (see chunking.Chunk.chunk_id).
Chroma's `upsert` then means re-running ingestion overwrites existing
chunks instead of duplicating them, which is one of the two acceptable
fixes the guide suggests for the Stage 5 watch-out.
"""

from typing import List

import chromadb

from .config import CHROMA_PERSIST_DIR, COLLECTION_NAME
from .chunking import Chunk
from .embeddings import embed_texts


def get_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))
    return client.get_or_create_collection(name=COLLECTION_NAME)


def index_chunks(chunks: List[Chunk]) -> int:
    """Embed and store chunks. Safe to re-run — upsert avoids duplicates."""
    if not chunks:
        return 0

    collection = get_collection()
    texts = [c.text for c in chunks]
    embeddings = embed_texts(texts)

    ids = [c.chunk_id for c in chunks]
    metadatas = [
        {"file_name": c.file_name, "page_number": c.page_number, "quarter": c.quarter}
        for c in chunks
    ]

    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
    )

    return collection.count()


def clear_collection():
    """Full reset — the other acceptable fix for duplicate ingestion."""
    client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))
    try:
        client.delete_collection(name=COLLECTION_NAME)
    except Exception:
        pass


def collection_stats() -> dict:
    collection = get_collection()
    return {
        "collection_name": COLLECTION_NAME,
        "persist_dir": str(CHROMA_PERSIST_DIR),
        "chunk_count": collection.count(),
    }
