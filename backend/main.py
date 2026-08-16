"""
Stage 10 (bonus) — FastAPI backend.

Three endpoints, exactly as specified:
  POST /index    — accepts uploaded PDFs, indexes them, returns file/chunk counts
  POST /ask      — accepts a question (+ optional top_k), returns answer + sources
  GET  /stats    — collection name, chunk count, models in use

Run with:  uvicorn backend.main:app --reload
Then test everything from /docs before touching app.py.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

from src.config import DATA_DIR, EMBEDDING_MODEL, CHAT_MODEL, DEFAULT_TOP_K
from src.rag_pipeline import ingest, ask as ask_pipeline, stats as pipeline_stats

app = FastAPI(title="Finance RAG Backend", version="1.0")


class AskRequest(BaseModel):
    question: str
    top_k: int = DEFAULT_TOP_K


class SourceOut(BaseModel):
    file_name: str
    page_number: int
    quarter: str
    distance: float


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceOut]


class IndexResponse(BaseModel):
    files_processed: int
    chunks_indexed: int
    total_chunks_in_store: int


class StatsResponse(BaseModel):
    collection_name: str
    persist_dir: str
    chunk_count: int
    embedding_model: str
    chat_model: str


@app.post("/index", response_model=IndexResponse)
async def index_endpoint(files: list[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    DATA_DIR.mkdir(exist_ok=True)
    for f in files:
        content = await f.read()
        (DATA_DIR / f.filename).write_bytes(content)

    try:
        result = ingest()
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e))

    return IndexResponse(
        files_processed=result.files_processed,
        chunks_indexed=result.chunks_indexed,
        total_chunks_in_store=result.total_chunks_in_store,
    )


@app.post("/ask", response_model=AskResponse)
async def ask_endpoint(req: AskRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    result = ask_pipeline(req.question, top_k=req.top_k)
    return AskResponse(
        answer=result.answer,
        sources=[
            SourceOut(
                file_name=s.file_name,
                page_number=s.page_number,
                quarter=s.quarter,
                distance=s.distance,
            )
            for s in result.sources
        ],
    )


@app.get("/stats", response_model=StatsResponse)
async def stats_endpoint():
    s = pipeline_stats()
    return StatsResponse(
        collection_name=s["collection_name"],
        persist_dir=s["persist_dir"],
        chunk_count=s["chunk_count"],
        embedding_model=EMBEDDING_MODEL,
        chat_model=CHAT_MODEL,
    )
