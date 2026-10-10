"""
app/services/quote_verifier.py
────────────────────────────────
The core "verified quotes" requirement: before any quote the model produces
is shown to the user, this module confirms it actually occurs in the
document's extracted text — independently of anything the model claims
about where it came from.

Design
──────
  • We never trust a position/page/offset the AI reports. We re-locate the
    quote ourselves by searching the raw extracted text.
  • Text extraction adds/removes whitespace (hyphenation, column wrapping,
    line breaks) without changing the actual words, so a naive
    `quote in document_text` exact-string check would wrongly reject quotes
    that are genuinely present. We normalize: split the quote into word
    tokens and search for those tokens separated by *any* run of
    whitespace, so "the Company shall\nindemnify" and "the Company shall
    indemnify" both match the same underlying words.
  • This is NOT fuzzy/approximate matching — every word must appear, in
    order, verbatim. A paraphrase does not verify. Only whitespace is
    elastic, exactly per the spec.
  • A case-only mismatch (rare — usually an OCR/extraction artifact) is
    still accepted as verified but flagged, rather than rejected outright;
    the words are genuinely present, just differently cased in the source.
"""

import re
from dataclasses import dataclass, field


@dataclass
class QuoteSpan:
    start: int  # char offset into the document's raw extracted text
    end: int    # exclusive


@dataclass
class VerifiedQuote:
    quote: str                          # verbatim text as it appears in the source (not the model's wording)
    verified: bool
    spans: list[QuoteSpan] = field(default_factory=list)   # every occurrence found
    case_mismatch: bool = False
    page: int | None = None             # page of the first occurrence, if page map available
    page_end: int | None = None         # last page the first occurrence spans into, if different from `page`


def _build_pattern(quote: str) -> "re.Pattern | None":
    tokens = quote.split()
    if not tokens:
        return None
    # Each token escaped literally; tokens joined by "any amount of
    # whitespace" so line-wrap/hyphenation artifacts in extraction don't
    # break an otherwise-genuine match. No nested quantifiers → no
    # catastrophic-backtracking risk regardless of quote length.
    body = r"\s+".join(re.escape(t) for t in tokens)
    try:
        return re.compile(body)
    except re.error:
        return None


def locate_page(char_offset: int, pages: list[dict]) -> int | None:
    """pages: list of {"page_number": int, "char_start": int, "char_end": int}."""
    for p in pages:
        if p["char_start"] <= char_offset < p["char_end"]:
            return p["page_number"]
    return None


def verify_quote(quote: str, document_text: str, pages: list[dict] | None = None) -> VerifiedQuote:
    """
    Search `document_text` for `quote`, tolerant of whitespace differences.

    Returns a VerifiedQuote with every occurrence found (citation
    highlighting needs to know about repeats, not just the first hit).
    """
    quote = quote.strip()
    if not quote:
        return VerifiedQuote(quote=quote, verified=False)

    pattern = _build_pattern(quote)
    if pattern is None:
        return VerifiedQuote(quote=quote, verified=False)

    matches = list(pattern.finditer(document_text))
    case_mismatch = False
    if not matches:
        # Fall back to a case-insensitive pass — the words are the same,
        # just differently cased (common after OCR or extraction).
        ci_pattern = re.compile(pattern.pattern, re.IGNORECASE)
        matches = list(ci_pattern.finditer(document_text))
        case_mismatch = bool(matches)

    if not matches:
        return VerifiedQuote(quote=quote, verified=False)

    spans = [QuoteSpan(m.start(), m.end()) for m in matches]
    verbatim = document_text[spans[0].start:spans[0].end]
    page = locate_page(spans[0].start, pages) if pages else None
    # end - 1: char_end is exclusive, so the span's last actual character is
    # one before it — using char_end itself would attribute a quote that
    # ends exactly on a page boundary to the page AFTER it, when every
    # character of the quote is actually on the earlier page.
    page_end = locate_page(spans[0].end - 1, pages) if pages else None

    return VerifiedQuote(
        quote=verbatim,
        verified=True,
        spans=spans,
        case_mismatch=case_mismatch,
        page=page,
        page_end=page_end if page_end != page else None,
    )


def verify_quotes(quotes: list[str], document_text: str, pages: list[dict] | None = None) -> list[VerifiedQuote]:
    return [verify_quote(q, document_text, pages) for q in quotes]
