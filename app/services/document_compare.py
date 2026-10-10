"""
app/services/document_compare.py
───────────────────────────────────
Part B #7: document comparison.

Diffs two document versions at clause/paragraph level (difflib.SequenceMatcher
over a list of segments, not characters — a single word changing inside a
2000-character clause doesn't fragment into dozens of tiny diff hunks), then
asks the model to classify each changed pair in plain language and rate how
significant the change is — explicitly distinguishing a reworded sentence
from a substantive change (e.g. a liability cap moving from one figure to
another), per the spec's own example.
"""

import difflib
import json
import logging

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from app.services.langgraph_rag import invoke_with_retry
from app.services.clause_detection import detect_clauses

logger = logging.getLogger("document_compare")

SIGNIFICANCE_LEVELS = ("minor", "moderate", "major")


def _segments(text: str) -> list[str]:
    """Split into clause-level segments using detected headings when
    possible — confirmed live that PDF extraction (PyMuPDF's
    get_text("text")) puts a single "\\n" between essentially every line
    and never a blank line between paragraphs, so a naive "\\n\\n" split
    collapsed an entire multi-clause contract into one or two "paragraphs",
    diffing the whole document as a single giant changed block instead of
    clause-level changes. Falls back to blank-line paragraphs (python-docx
    extraction genuinely does join paragraphs with "\\n\\n" — see
    docx_loader.py) or, as a last resort, individual non-empty lines, for a
    document with no detectable heading structure at all.
    """
    clauses = detect_clauses(text)
    if len(clauses) >= 2:
        segments = [text[c["char_start"]:c["char_end"]].strip() for c in clauses]
        return [s for s in segments if s]

    blank_line_paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    if len(blank_line_paras) >= 2:
        return blank_line_paras

    return [line.strip() for line in text.split("\n") if line.strip()]


def _classify_changes(pairs: list[dict], llm) -> list[dict]:
    """One batched LLM call grading every changed-paragraph pair at once —
    same reasoning as grade_documents() in langgraph_rag.py: N separate
    calls for N changes doesn't scale to a real contract's worth of diffs.
    """
    if not pairs:
        return []

    numbered = "\n\n".join(
        f"[{i + 1}]\nBEFORE: {p['before'] or '(not present)'}\nAFTER: {p['after'] or '(not present)'}"
        for i, p in enumerate(pairs)
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You compare BEFORE/AFTER contract paragraph pairs. For EACH numbered pair, "
         "write a short plain-language description of what changed IN SUBSTANCE (not "
         "phrasing), and rate its significance:\n"
         "  - 'minor': rewording, formatting, typo fixes, renumbering — meaning unchanged\n"
         "  - 'moderate': a real but limited change (e.g. a notice period, a defined term)\n"
         "  - 'major': a change that shifts rights, obligations, money, liability, or risk "
         "(e.g. a liability cap, a payment amount, a termination right, an indemnity)\n"
         "A rewritten sentence that doesn't change the deal is 'minor', even if every word "
         "is different. A single-number change to a cap, fee, or deadline is at least "
         "'moderate', usually 'major'.\n\n"
         "Output ONLY a JSON array, one object per pair in order, each exactly "
         '{{"summary": "...", "significance": "minor"|"moderate"|"major"}}. No other text.'),
        ("human", "{pairs}"),
    ])
    chain = prompt | llm | StrOutputParser()
    raw = invoke_with_retry(chain, {"pairs": numbered})

    try:
        start, end = raw.index("["), raw.rindex("]") + 1
        parsed = json.loads(raw[start:end])
    except (ValueError, json.JSONDecodeError) as exc:
        logger.warning("[compare] Couldn't parse significance grading, defaulting to 'moderate': %s raw=%r", exc, raw[:300])
        parsed = []

    results = []
    for i in range(len(pairs)):
        if i < len(parsed) and isinstance(parsed[i], dict):
            sig = parsed[i].get("significance", "moderate")
            if sig not in SIGNIFICANCE_LEVELS:
                sig = "moderate"
            results.append({"summary": parsed[i].get("summary", "Changed."), "significance": sig})
        else:
            results.append({"summary": "Content changed between versions.", "significance": "moderate"})
    return results


def compare_documents(text_a: str, filename_a: str, text_b: str, filename_b: str, llm) -> dict:
    """
    Returns {"changes": [...]} — each change:
      {type: "added"|"removed"|"modified", before, after, summary, significance}
    sorted major → minor so the most substantive changes surface first.
    """
    paras_a = _segments(text_a)
    paras_b = _segments(text_b)

    matcher = difflib.SequenceMatcher(None, paras_a, paras_b, autojunk=False)
    opcodes = matcher.get_opcodes()

    changes: list[dict] = []
    modified_pairs: list[dict] = []
    modified_indices: list[int] = []

    for tag, i1, i2, j1, j2 in opcodes:
        if tag == "equal":
            continue
        if tag == "replace":
            # Pair up 1:1 as far as both sides go; leftover on the longer
            # side becomes its own added/removed entries.
            before_block = paras_a[i1:i2]
            after_block = paras_b[j1:j2]
            for k in range(max(len(before_block), len(after_block))):
                before = before_block[k] if k < len(before_block) else ""
                after = after_block[k] if k < len(after_block) else ""
                if before and after:
                    modified_indices.append(len(changes))
                    modified_pairs.append({"before": before, "after": after})
                    changes.append({"type": "modified", "before": before, "after": after})
                elif after:
                    changes.append({"type": "added", "before": "", "after": after,
                                     "summary": "New clause added.", "significance": "moderate"})
                else:
                    changes.append({"type": "removed", "before": before, "after": "",
                                     "summary": "Clause removed.", "significance": "moderate"})
        elif tag == "insert":
            for after in paras_b[j1:j2]:
                changes.append({"type": "added", "before": "", "after": after,
                                 "summary": "New clause added.", "significance": "moderate"})
        elif tag == "delete":
            for before in paras_a[i1:i2]:
                changes.append({"type": "removed", "before": before, "after": "",
                                 "summary": "Clause removed.", "significance": "moderate"})

    # Grade the true modifications (added/removed get a generic label above —
    # presence/absence is already the substantive signal there).
    try:
        graded = _classify_changes(modified_pairs, llm)
        for idx, g in zip(modified_indices, graded):
            changes[idx]["summary"] = g["summary"]
            changes[idx]["significance"] = g["significance"]
    except Exception as exc:
        logger.warning("[compare] Significance grading failed, leaving defaults: %s", exc)
        for idx in modified_indices:
            changes[idx].setdefault("summary", "Content changed between versions.")
            changes[idx].setdefault("significance", "moderate")

    order = {"major": 0, "moderate": 1, "minor": 2}
    changes.sort(key=lambda c: order.get(c["significance"], 1))

    return {
        "document_a": filename_a,
        "document_b": filename_b,
        "total_segments_a": len(paras_a),
        "total_segments_b": len(paras_b),
        "changes": changes,
    }
