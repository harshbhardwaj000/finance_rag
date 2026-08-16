"""
Stage 6 — Retrieval.

"Almost every mark lost in this assignment is lost at the clerk stage,
not the intern stage." retrieve() is deliberately the one function you
should call and print the output of constantly while debugging —
see the guide's habit box.
"""

from dataclasses import dataclass
from typing import List

from .config import DEFAULT_TOP_K
from .embeddings import embed_query
from .vectorstore import get_collection


@dataclass
class RetrievedChunk:
    text: str
    file_name: str
    page_number: int
    quarter: str
    distance: float


def retrieve(question: str, top_k: int = DEFAULT_TOP_K) -> List[RetrievedChunk]:
    collection = get_collection()
    query_embedding = embed_query(question)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
    )

    retrieved: List[RetrievedChunk] = []
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]

    for doc, meta, dist in zip(docs, metas, dists):
        retrieved.append(
            RetrievedChunk(
                text=doc,
                file_name=meta.get("file_name", "?"),
                page_number=meta.get("page_number", -1),
                quarter=meta.get("quarter", "?"),
                distance=dist,
            )
        )

    return retrieved


if __name__ == "__main__":
    import sys

    q = sys.argv[1] if len(sys.argv) > 1 else "What was revenue in the latest quarter?"
    for r in retrieve(q):
        print(f"[{r.file_name} p.{r.page_number} | {r.quarter} | dist={r.distance:.4f}]")
        print(r.text[:300])
        print("-" * 60)
