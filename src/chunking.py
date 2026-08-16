"""
Stage 3 — Chunking.

Recursive character splitting, 800-1200 chars, 100-200 overlap (see
config.py for the chosen values and why).

Stage 6 of the guide identifies the defining failure mode of this
assignment: four quarterly reports from the same company reuse almost
identical sentences ("Revenue grew during the quarter..."), so a chunk
about Q1 and a chunk about Q3 can look equally relevant to a Q3
question. The fix applied here is to prefix every chunk's *text* (not
just its metadata) with its source label before embedding, so the
quarter becomes part of what similarity search is matching on.
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import CHUNK_SIZE, CHUNK_OVERLAP
from .pdf_extraction import PageText


@dataclass
class Chunk:
    chunk_id: str
    text: str  # includes the source-label prefix — this is what gets embedded
    file_name: str
    page_number: int
    quarter: str


_QUARTER_PATTERN = re.compile(r"(Q[1-4][_\-\s]?FY?\d{2,4})", re.IGNORECASE)


def guess_quarter(file_name: str) -> str:
    """Best-effort quarter label from a filename like Infosys_Q1_FY26.pdf."""
    match = _QUARTER_PATTERN.search(file_name)
    if match:
        return match.group(1).replace("_", " ").replace("-", " ").upper()
    return Path(file_name).stem.replace("_", " ")


def chunk_pages(pages: List[PageText]) -> List[Chunk]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: List[Chunk] = []
    for page in pages:
        if not page.text.strip():
            continue

        quarter = guess_quarter(page.file_name)
        pieces = splitter.split_text(page.text)

        for i, piece in enumerate(pieces):
            source_label = f"[Source: {page.file_name}, {quarter}, page {page.page_number}]\n"
            chunk_text = source_label + piece
            chunk_id = f"{Path(page.file_name).stem}_p{page.page_number}_c{i}"
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    text=chunk_text,
                    file_name=page.file_name,
                    page_number=page.page_number,
                    quarter=quarter,
                )
            )

    return chunks


if __name__ == "__main__":
    from .config import DATA_DIR
    from .pdf_extraction import extract_all

    pages = extract_all(DATA_DIR)
    chunks = chunk_pages(pages)
    print(f"Total chunks: {len(chunks)}")
    import random

    for c in random.sample(chunks, min(3, len(chunks))):
        print("=" * 60)
        print(c.chunk_id)
        print(c.text[:400])
