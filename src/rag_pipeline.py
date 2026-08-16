"""
Ties Stages 2-8 together into two calls: ingest() and ask().
Both the Streamlit app and the FastAPI backend (Stage 10) call only this
module — neither one re-implements chunking or prompting itself, which
is exactly the "separate the brain from the face" split Stage 10 wants.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from .config import DATA_DIR, DEFAULT_TOP_K
from .pdf_extraction import extract_all
from .chunking import chunk_pages, Chunk
from .vectorstore import index_chunks, collection_stats, clear_collection
from .retrieval import retrieve, RetrievedChunk
from .prompt import answer_question


@dataclass
class IngestResult:
    files_processed: int
    chunks_indexed: int
    total_chunks_in_store: int


@dataclass
class AskResult:
    answer: str
    sources: List[RetrievedChunk] = field(default_factory=list)


def ingest(data_dir: Path = DATA_DIR, reset: bool = False) -> IngestResult:
    """Stages 2-5: extract -> chunk -> embed -> store. Safe to re-run (upsert)."""
    if reset:
        clear_collection()

    pages = extract_all(data_dir)
    chunks: List[Chunk] = chunk_pages(pages)
    total = index_chunks(chunks)

    file_count = len({p.file_name for p in pages})
    return IngestResult(
        files_processed=file_count,
        chunks_indexed=len(chunks),
        total_chunks_in_store=total,
    )


def ask(question: str, top_k: int = DEFAULT_TOP_K) -> AskResult:
    """Stages 6-8: retrieve -> prompt -> answer, with sources attached."""
    chunks = retrieve(question, top_k=top_k)
    if not chunks:
        return AskResult(
            answer="No documents have been indexed yet. Upload and index PDFs first.",
            sources=[],
        )

    answer = answer_question(question, chunks)
    return AskResult(answer=answer, sources=chunks)


def stats() -> dict:
    return collection_stats()
