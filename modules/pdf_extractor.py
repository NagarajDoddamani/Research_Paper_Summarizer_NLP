"""PDF parsing and user-facing validation errors."""
from __future__ import annotations

from dataclasses import dataclass

import pymupdf


class PDFExtractionError(ValueError):
    """An uploaded file cannot be read as a text-bearing PDF."""


@dataclass(frozen=True)
class PDFDocument:
    title: str
    page_count: int
    word_count: int
    pages: tuple[str, ...]
    text: str


def extract_pdf(file_bytes: bytes) -> PDFDocument:
    """Extract page-wise text and metadata from PDF bytes using PyMuPDF."""
    if not file_bytes:
        raise PDFExtractionError("The uploaded PDF is empty.")
    try:
        with pymupdf.open(stream=file_bytes, filetype="pdf") as doc:
            if doc.is_encrypted:
                raise PDFExtractionError("This PDF is encrypted. Upload an unlocked PDF.")
            if len(doc) == 0:
                raise PDFExtractionError("This PDF contains no pages.")
            pages = tuple(page.get_text("text").strip() for page in doc)
            text = "\n\n".join(part for part in pages if part)
            if not text.strip():
                raise PDFExtractionError(
                    "No selectable text was found. This may be a scanned PDF; OCR is not included in this version."
                )
            metadata = doc.metadata or {}
            title = (metadata.get("title") or "").strip() or "Untitled document"
            return PDFDocument(title, len(doc), len(text.split()), pages, text)
    except PDFExtractionError:
        raise
    except (pymupdf.FileDataError, RuntimeError, ValueError) as exc:
        raise PDFExtractionError("The file could not be read as a valid PDF. Check the file and try again.") from exc
