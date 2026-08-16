"""
Stage 4 — Turn chunks into embeddings.

text-embedding-3-small, sent in batches (not one call per chunk — the
guide is explicit that looping one-at-a-time is "slow enough to be
annoying"). The same model must be used for chunks and for questions;
embed_query() and embed_texts() both live here so that's true by
construction, not by convention.
"""

from typing import List

from openai import OpenAI

from .config import OPENAI_API_KEY, EMBEDDING_MODEL

_client = None

# def get_client() -> OpenAI:
#     global _client
#     if _client is None:
#         if not OPENAI_API_KEY:
#             raise RuntimeError(
#                 "OPENAI_API_KEY is not set. Add it to your .env file (see .env.example)."
#             )
#         _client = OpenAI(api_key=OPENAI_API_KEY)
#     return _client

def get_client() -> OpenAI:
    global _client
    if _client is None:
        if not OPENAI_API_KEY:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Add it to your .env file (see .env.example)."
            )
        _client = OpenAI(
            api_key=OPENAI_API_KEY,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/" 
        )
    return _client


def embed_texts(texts: List[str], batch_size: int = 100) -> List[List[float]]:
    """Embed a list of chunk texts in batches. Order-preserving."""
    client = get_client()
    all_embeddings: List[List[float]] = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        response = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        # API preserves input order in response.data
        all_embeddings.extend([item.embedding for item in response.data])

    return all_embeddings


def embed_query(text: str) -> List[float]:
    """Embed a single question with the SAME model used for chunks."""
    return embed_texts([text])[0]
