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

import io
import fitz  # PyMuPDF — fast, pure C, no external dependencies


def extract_text(file) -> str:
    """
    Extract markdown-formatted text from a PDF file-like object (in-memory).

    Parameters
    ----------
    file : file-like object
        A SpooledTemporaryFile, BytesIO, or any readable binary stream.

    Returns
    -------
    str
        Extracted text content from all pages joined as a single string.
    """
    # Read the entire binary content into memory
    raw_bytes = file.read() if not isinstance(file, (bytes, bytearray)) else file

    # Open from a bytes buffer — no disk write
    doc = fitz.open(stream=raw_bytes, filetype="pdf")

    pages_text = []
    for page in doc:
        pages_text.append(page.get_text("text"))

    doc.close()
    return "\n\n".join(pages_text)