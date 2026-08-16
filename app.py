"""
Stage 9 — Interface.

Five required pieces, per the guide's checkpoint:
  upload -> index (with file/chunk count feedback) -> ask -> answer -> sources
Plus: spinners during long operations, a guard against asking before
indexing, and a running history of previous Q&A pairs.
"""

import streamlit as st
from pathlib import Path

from src.config import DATA_DIR
from src.rag_pipeline import ingest, ask, stats

st.set_page_config(page_title="Finance RAG", page_icon="📊", layout="wide")
st.title("📊 Quarterly Report RAG Assistant")
st.caption(
    "Upload a few quarters of press releases for one company, index them, "
    "then ask questions grounded strictly in what's on the page."
)

if "history" not in st.session_state:
    st.session_state.history = []
if "indexed" not in st.session_state:
    # Treat any existing chunks in the persisted store as already indexed
    st.session_state.indexed = stats()["chunk_count"] > 0

# ---- Sidebar: store status -------------------------------------------------
with st.sidebar:
    st.subheader("Store status")
    s = stats()
    st.write(f"**Collection:** {s['collection_name']}")
    st.write(f"**Chunks stored:** {s['chunk_count']}")
    st.caption(f"Persisted to `{s['persist_dir']}`")

# ---- Upload + index ---------------------------------------------------------
st.subheader("1. Upload & index")

uploaded_files = st.file_uploader(
    "Upload quarterly report PDFs (press releases preferred)",
    type=["pdf"],
    accept_multiple_files=True,
)

col1, col2 = st.columns([1, 1])
with col1:
    reset_first = st.checkbox(
        "Clear existing store before indexing", value=False,
        help="Use this if you're re-indexing from scratch to avoid stale chunks.",
    )
with col2:
    index_clicked = st.button("Index uploaded files", type="primary")

if index_clicked:
    if uploaded_files:
        DATA_DIR.mkdir(exist_ok=True)
        for f in uploaded_files:
            (DATA_DIR / f.name).write_bytes(f.getbuffer())

    with st.spinner("Extracting text, chunking, embedding, and storing..."):
        try:
            result = ingest(reset=reset_first)
            st.session_state.indexed = True
            st.success(
                f"Indexed {result.files_processed} file(s) → "
                f"{result.chunks_indexed} chunks this run "
                f"({result.total_chunks_in_store} total chunks in store)."
            )
        except FileNotFoundError as e:
            st.error(str(e))
        except ValueError as e:
            st.error(str(e))

st.divider()

# ---- Ask ---------------------------------------------------------------
st.subheader("2. Ask a question")

if not st.session_state.indexed:
    st.info("Upload and index at least one PDF before asking questions.")

question = st.text_input(
    "Your question",
    placeholder="e.g. What was revenue in the latest quarter?",
    disabled=not st.session_state.indexed,
)
top_k = st.slider("How many chunks to retrieve (top_k)", 2, 8, 4)
ask_clicked = st.button(
    "Ask", disabled=not st.session_state.indexed or not question.strip()
)

if ask_clicked and question.strip():
    with st.spinner("Retrieving relevant chunks and generating an answer..."):
        result = ask(question, top_k=top_k)
    st.session_state.history.insert(0, {"question": question, "result": result})

# ---- Answer + sources + history ---------------------------------------------
st.subheader("3. Answers")

for entry in st.session_state.history:
    q = entry["question"]
    r = entry["result"]
    with st.container(border=True):
        st.markdown(f"**Q: {q}**")
        st.write(r.answer)
        if r.sources:
            with st.expander(f"Sources ({len(r.sources)})"):
                for s in r.sources:
                    st.markdown(
                        f"- `{s.file_name}` — page {s.page_number} — {s.quarter} "
                        f"(distance {s.distance:.3f})"
                    )
