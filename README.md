# Finance RAG — Quarterly Report Q&A

A retrieval-augmented Q&A system over a company's quarterly financial
press releases, built to the assignment brief's 12-stage guide.

## What's in here

```
finance_rag/
├── app.py                 # Streamlit interface (Stage 9)
├── backend/main.py        # Optional FastAPI backend, 3 endpoints (Stage 10, bonus)
├── src/
│   ├── config.py           # All tunable decisions in one place
│   ├── pdf_extraction.py   # Stage 2
│   ├── chunking.py         # Stage 3 (+ Stage 6 quarter-prefix fix)
│   ├── embeddings.py       # Stage 4
│   ├── vectorstore.py      # Stage 5 (Chroma, persistent, upsert-safe)
│   ├── retrieval.py        # Stage 6
│   ├── prompt.py           # Stage 7
│   └── rag_pipeline.py     # ties ingest()/ask() together for both frontends
├── data/                   # put your PDFs here
├── chroma_store/           # persisted vector DB (gitignored)
├── requirements.txt
└── .env.example
```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env            # then paste your real OPENAI_API_KEY into .env
```

## Get your documents (Stage 1)

Pick one listed company, download 3–4 consecutive quarters of press
releases from its Investor Relations page, and save them into `data/`
named like `CompanyName_Q1_FY26.pdf` (the quarter pattern in the
filename is used to tag chunks — see Stage 6 below).

**Before you accept a file:** open it and try to highlight a line of
text. If nothing highlights, it's a scanned image and extraction will
return empty text — `pdf_extraction.py` will raise a clear error
telling you which file failed this check.

## Run it

**Streamlit (required):**
```bash
streamlit run app.py
```
Upload your PDFs in the UI, click "Index uploaded files", then ask
questions. Sources (file name + page number) appear under every answer.

**FastAPI backend (bonus, optional):**
```bash
uvicorn backend.main:app --reload
```
Open `http://127.0.0.1:8000/docs` and test `/index`, `/ask`, `/stats`
directly before wiring a frontend to it.

**Command-line checks**, useful while debugging each stage in isolation:
```bash
python -m src.pdf_extraction     # prints page counts + first 300 chars/file
python -m src.chunking           # prints total chunk count + 3 random chunks
python -m src.retrieval "What was revenue in the latest quarter?"
```

## Key decisions (fill these in for your submission)

| Decision | Value | Why |
|---|---|---|
| Chunk size | 1200 chars (`src/config.py`) | keeps full tables inside one chunk |
| Overlap | 150 chars | prevents boundary sentences being split |
| top_k | 4 (adjustable in the UI) | — |
| Embedding model | text-embedding-3-small | same model for chunks & queries |
| Chat model | gpt-4o | temperature 0.2 for consistent factual answers |
| Duplicate-ingestion fix | stable chunk IDs + `upsert` | re-running ingestion overwrites, doesn't duplicate |
| Same-wording-across-quarters fix | quarter label prefixed into chunk *text* before embedding (Stage 6) | otherwise Q1 and Q3 chunks look equally relevant |

**Company chosen:** _fill in_
**Source PDF links:** _fill in_

## Test results (Stage 11)

Fill in the 10-question table from the brief here, including the ones
that came out wrong — a diagnosed failure is worth more than a hidden one.

| # | Question | Correct? | If wrong, what did retrieval return? |
|---|---|---|---|
| 1 | Revenue in the latest quarter | | |
| 2 | Net profit compared across quarters | | |
| 3 | Year-on-year revenue comparison | | |
| 4 | Management commentary on demand | | |
| 5 | Fastest-growing segment | | |
| 6 | Operating margin trend | | |
| 7 | Dividend declared | | |
| 8 | Risks and headwinds | | |
| 9 | Three-line summary | | |
| 10 | Trap question (must be refused) | | |

## What didn't work

_Fill in honestly — e.g. "question 6 failed because operating margin
appears only inside a table split across two chunks."_
