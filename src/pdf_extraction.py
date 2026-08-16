"""
Stage 2 — Read the PDFs into text.

The one rule that matters: every piece of extracted text keeps its
source file name and page number attached. Losing that here means you
cannot show citations later, and the guide is explicit that this is
almost impossible to bolt back on afterwards.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List

import pdfplumber


@dataclass
class PageText:
    file_name: str
    page_number: int  # 1-indexed, human-friendly
    text: str


def extract_pdf(path: Path) -> List[PageText]:
    """Extract text page-by-page from a single PDF, tagged with source info."""
    pages: List[PageText] = []
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            raw = page.extract_text() or ""
            pages.append(PageText(file_name=path.name, page_number=i, text=raw))
    return pages


def extract_all(data_dir: Path) -> List[PageText]:
    """Extract every PDF in a directory. Skips non-PDF files."""
    all_pages: List[PageText] = []
    pdf_files = sorted(data_dir.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(
            f"No PDFs found in {data_dir}. Add your quarterly report PDFs there first."
        )

    for pdf_path in pdf_files:
        pages = extract_pdf(pdf_path)
        empty = sum(1 for p in pages if not p.text.strip())
        if empty == len(pages):
            # Guide's "selection test" failure mode, caught in code: a scanned
            # PDF with no selectable text produces blank extraction for every page.
            raise ValueError(
                f"{pdf_path.name}: extracted 0 characters from all {len(pages)} "
                "pages. This is almost always a scanned image PDF — re-check "
                "with the text-selection test (Stage 1) or OCR it first."
            )
        all_pages.extend(pages)

    return all_pages


if __name__ == "__main__":
    # Quick manual check, matching the guide's Stage 2 checkpoint:
    # print page count per file and the first 300 chars of page 1.
    from .config import DATA_DIR

    pages = extract_all(DATA_DIR)
    by_file = {}
    for p in pages:
        by_file.setdefault(p.file_name, []).append(p)

    for fname, fpages in by_file.items():
        print(f"{fname}: {len(fpages)} pages")
        print(fpages[0].text[:300])
        print("---")
