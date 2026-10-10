"""
app/services/clause_detection.py
───────────────────────────────────
Best-effort contract heading detection — not a real document-structure
parser, just regex over likely heading shapes ("1. Definitions",
"Section 2:", an ALL-CAPS line). Shared by:
  • agent_research.py — gives the Part C agent a navigable table of
    contents (list_clauses / get_section tools).
  • document_compare.py — segments two documents at clause boundaries
    instead of naive "\\n\\n" splitting, which doesn't work: PDF
    extraction (PyMuPDF's get_text("text")) puts a single "\\n" between
    essentially every line, never a blank line between paragraphs, so
    splitting on "\\n\\n" collapses an entire multi-page contract into
    one or two "paragraphs" — confirmed live, where a document that
    genuinely has 10 distinct clauses diffed as a single giant changed
    block instead of clause-level changes.
"""

import re

_HEADING_PATTERNS = [
    re.compile(r"^\s*\d+(\.\d+)*[\.\)]\s+\S.{2,100}$"),                               # "1. Definitions" / "2.1) Payment Terms"
    re.compile(r"^\s*(Section|Article|Clause)\s+\d+[A-Za-z]?\s*[:.\-]?\s*.{0,100}$", re.IGNORECASE),
    re.compile(r"^\s*[A-Z][A-Z \-/&',]{4,80}\s*$"),                                   # ALL CAPS short heading line
]


def detect_clauses(text: str) -> list[dict]:
    """Returns [{"heading": str, "char_start": int, "char_end": int}, ...]
    in document order. Not exhaustive or guaranteed accurate — a heuristic,
    not a real parser — but good enough to give both the agent and the
    comparison feature real structure to work with instead of raw text.
    """
    lines = text.split("\n")
    clauses: list[dict] = []
    offset = 0
    for line in lines:
        stripped = line.strip()
        if stripped and len(stripped) < 120:
            for pat in _HEADING_PATTERNS:
                if pat.match(stripped):
                    clauses.append({"heading": stripped, "char_start": offset})
                    break
        offset += len(line) + 1  # +1 for the "\n" split boundary
    for i, c in enumerate(clauses):
        c["char_end"] = clauses[i + 1]["char_start"] if i + 1 < len(clauses) else len(text)
    return clauses
