from __future__ import annotations

from typing import List

from pypdf import PdfReader


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extracts text from a PDF file using pypdf.

    Returns a single string with pages separated by blank lines.
    """
    page_texts: List[str] = []
    with open(pdf_path, "rb") as fp:
        reader = PdfReader(fp)
        for page in reader.pages:
            try:
                text = page.extract_text() or ""
            except Exception:
                text = ""
            page_texts.append(text.strip())

    # Keep simple joining; downstream summarizer handles empty/short content
    return "\n\n".join(t for t in page_texts if t)



