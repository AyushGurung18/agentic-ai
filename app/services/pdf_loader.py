"""
app/services/pdf_loader.py
──────────────────────────
Extracts plain text from a PDF file-like object entirely in memory.

Why in-memory?
  • HF Spaces containers run with a read-only filesystem outside /tmp and /data.
  • Writing temp files risks exceeding ephemeral storage or leaving orphans.
  • pymupdf (fitz) supports opening PDFs from a bytes buffer directly, so we
    never touch the disk at all.
"""

from dataclasses import dataclass, field

import fitz  # PyMuPDF — fast, pure C, no external dependencies


@dataclass
class PageOffset:
    page_number: int   # 1-indexed, matches what a human calls "page N"
    char_start: int     # inclusive offset into the joined text
    char_end: int        # exclusive offset into the joined text


@dataclass
class ExtractResult:
    text: str
    pages: list[PageOffset] = field(default_factory=list)

    @property
    def page_count(self) -> int:
        return len(self.pages)


def extract_text(file) -> str:
    """Backward-compatible wrapper — returns just the joined text."""
    return extract_pdf(file).text


def extract_pdf(file) -> ExtractResult:
    """
    Extract text from a PDF file-like object (in-memory), tracking the
    character offset range of each page within the joined text.

    The offsets are what citation highlighting uses to turn a verified
    quote's character range (found by searching the raw text) into "this is
    on page 7" without ever trusting a page number the AI itself reports.

    Parameters
    ----------
    file : file-like object
        A SpooledTemporaryFile, BytesIO, or any readable binary stream.

    Returns
    -------
    ExtractResult
        .text  — all pages joined with "\\n\\n"
        .pages — per-page PageOffset(page_number, char_start, char_end)
    """
    raw_bytes = file.read() if not isinstance(file, (bytes, bytearray)) else file

    doc = fitz.open(stream=raw_bytes, filetype="pdf")

    pages_text = []
    offsets: list[PageOffset] = []
    cursor = 0
    for i, page in enumerate(doc):
        page_text = page.get_text("text")
        start = cursor
        end = start + len(page_text)
        offsets.append(PageOffset(page_number=i + 1, char_start=start, char_end=end))
        pages_text.append(page_text)
        # +2 for the "\n\n" joiner added below, so offsets stay accurate
        # against the actual joined string.
        cursor = end + 2

    doc.close()
    return ExtractResult(text="\n\n".join(pages_text), pages=offsets)
