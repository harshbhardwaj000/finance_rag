"""
Central configuration for the Finance RAG system.

Every "decision" the assignment guide asks you to write down and justify
lives here as a single named constant, so your README can just point at
this file instead of hunting through the codebase.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ---- API ---------------------------------------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    # Don't crash at import time (e.g. during pytest collection) — but every
    # real code path that needs the key will fail loudly and immediately.
    pass

# ---- Models --------------------------------------------------------------
# Guide Stage 4: chunks and questions MUST be embedded by the same model.
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o"
TEMPERATURE = 0.2  # Guide Stage 7: 0 or 0.2, low = consistent factual answers

# ---- Chunking (Guide Stage 3) --------------------------------------------
# Financial press releases are mostly tables; the guide recommends trying
# 800 vs 1200 and keeping whichever keeps a full table inside one chunk.
# Default here is the upper end, which the guide notes usually wins.
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 150

# ---- Retrieval (Guide Stage 6) --------------------------------------------
DEFAULT_TOP_K = 4

# ---- Paths -----------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CHROMA_PERSIST_DIR = BASE_DIR / "chroma_store"
COLLECTION_NAME = "finance_reports"

DATA_DIR.mkdir(exist_ok=True)
CHROMA_PERSIST_DIR.mkdir(exist_ok=True)
