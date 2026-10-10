"""
app/services/quote_annotator.py
──────────────────────────────────
Bridges the model's output (which wraps verbatim quotes in QUOTE_MARK per
the instructions in langgraph_rag.generate()) to quote_verifier.py, and
produces:

  1. The rewritten answer text — same QUOTE_MARK-delimited shape, but with
     the model's quote text replaced by the verbatim source text when a
     match is found (guarantees the text shown to the user is byte-identical
     to what's highlightable in the source document).
  2. An ordered list of metadata dicts, one per quote span in the order it
     appears in the text — the frontend walks QUOTE_MARK pairs in order and
     looks up this list by index to know verified/unverified, which
     document, which page, and the char range to highlight.

Used identically by the standard chat pipeline (langgraph_rag.py) and the
Part C agentic-research pipeline (agent_research.py) — "verified quotes
still apply to the final answer" applies to both the same way.
"""

import re
from dataclasses import dataclass

from app.services.quote_verifier import verify_quote

QUOTE_MARK = "\x1d"
# Document-index marker, e.g. "\x1c2\x1c" immediately before a quote span —
# a separate control character outside the QUOTE_MARK pair entirely, never
# inside it. Earlier versions tried parsing the index as a text prefix
# *inside* the quote ("2:text" or "[2]text"), which is fundamentally
# ambiguous: any prefix shape can collide with genuine document text that
# happens to start the same way (a clause literally numbered "2:", a time
# like "10:00 AM", a clause bracket-numbered "[2] Indemnification..."). A
# structurally separate delimiter removes the collision class entirely —
# the quote-content group can never be confused with the index token
# because the parser never looks for the index pattern *inside* it.
DOC_INDEX_MARK = "\x1c"
_QUOTE_PATTERN = re.compile(
    r"(?:" + re.escape(DOC_INDEX_MARK) + r"(\d+)" + re.escape(DOC_INDEX_MARK) + r")?"
    + re.escape(QUOTE_MARK) + r"(.*?)" + re.escape(QUOTE_MARK),
    re.DOTALL,
)


@dataclass
class DocRef:
    document_id: str
    filename: str
    text: str
    pages: list[dict]


def annotate_quotes(generation: str, doc_refs: dict[int, DocRef]) -> tuple[str, list[dict]]:
    """
    Parameters
    ----------
    generation : raw model output containing QUOTE_MARK-delimited spans,
                 optionally preceded by a DOC_INDEX_MARK-delimited document
                 index (1-based, matching the [Document N: ...] labels
                 generate() put in the prompt) — see DOC_INDEX_MARK above.
    doc_refs   : {1: DocRef(...), 2: DocRef(...), ...} — the documents that
                 were actually in context for this generation.

    Returns
    -------
    (rewritten_text, quote_metadata) — quote_metadata[i] describes the i-th
    QUOTE_MARK pair in rewritten_text, in order.
    """
    metadata: list[dict] = []

    if not doc_refs:
        # No documents were in context at all (e.g. pure web-search fallback,
        # or a session with nothing attached) — nothing to verify against.
        # Strip any quote markers the model still produced so stray, unverifiable
        # claims don't appear as confidently "wrapped" with no content.
        def _strip(m):
            return m.group(2)
        return _QUOTE_PATTERN.sub(_strip, generation), metadata

    def _replace(m: re.Match) -> str:
        index_group, quote_text = m.group(1), m.group(2)
        doc_idx = int(index_group) if index_group and int(index_group) in doc_refs else 1

        ref = doc_refs.get(doc_idx) or next(iter(doc_refs.values()))
        quote_text = quote_text.strip()
        vq = verify_quote(quote_text, ref.text, ref.pages)

        metadata.append({
            "verified": vq.verified,
            "document_id": ref.document_id,
            "filename": ref.filename,
            "page": vq.page,
            "occurrences": len(vq.spans),
            "char_start": vq.spans[0].start if vq.spans else None,
            "char_end": vq.spans[0].end if vq.spans else None,
            "case_mismatch": vq.case_mismatch,
        })

        shown_text = vq.quote if vq.verified else quote_text
        return f"{QUOTE_MARK}{shown_text}{QUOTE_MARK}"

    rewritten = _QUOTE_PATTERN.sub(_replace, generation)
    return rewritten, metadata
