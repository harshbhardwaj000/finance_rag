"""
Stage 7 — The instruction you give GPT-4o.

Two non-negotiable rules baked into SYSTEM_PROMPT:
  1. Answer only from the provided context.
  2. If the context doesn't contain the answer, say so plainly.
Figures must carry their unit and period (₹41,000 crore, Q1 FY26 — not
just "41,000"), so a fact can't survive stripped of the context that
makes it checkable.
"""

from typing import List

from .config import CHAT_MODEL, TEMPERATURE
from .embeddings import get_client
from .retrieval import RetrievedChunk

SYSTEM_PROMPT = """You are a financial analyst assistant. You answer questions about a \
company's quarterly results using ONLY the context provided below — never your own \
knowledge or training data about this or any other company.

Rules you must follow exactly:
1. Answer only using facts that appear in the provided context chunks. Do not add, \
infer, or recall any figure that is not explicitly present in the context.
2. If the context does not contain enough information to answer the question, say so \
plainly — e.g. "The provided documents do not contain this information." Do not guess.
3. Every number you state must include its unit and its period/quarter exactly as given \
in the context (e.g. "₹41,000 crore for Q1 FY26", not just "41,000").
4. Be concise and direct. Do not pad the answer with disclaimers beyond what is required \
by rule 2.
"""


def build_context(chunks: List[RetrievedChunk]) -> str:
    parts = []
    for i, c in enumerate(chunks, start=1):
        parts.append(f"--- Chunk {i} ---\n{c.text}")
    return "\n\n".join(parts)


def answer_question(question: str, chunks: List[RetrievedChunk]) -> str:
    context = build_context(chunks)

    user_prompt = f"""Context:
{context}

Question: {question}

Answer using only the context above."""

    client = get_client()
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=TEMPERATURE,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.choices[0].message.content
