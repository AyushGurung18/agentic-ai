"""
app/services/agent_research.py
─────────────────────────────────
Part C, Option 2: agentic document research.

Instead of stuffing retrieved chunks into one prompt, the model is given
tools — search_document, get_section, list_clauses — and decides for itself
what to look up before answering, across multiple rounds.

Why a prompted protocol instead of native tool-calling
────────────────────────────────────────────────────────
The chat LLM here is a 5-6 provider fallback chain (Groq/Gemini/NVIDIA/
OpenRouter/self-hosted vLLM — see rag.get_llm_by_intent) chosen specifically
so the app keeps working when any one provider is down or rate-limited.
Native function-calling support is inconsistent across that chain (vLLM in
particular only supports it with a specific server-side flag + parser that
may not be enabled), and LangChain's `.with_fallbacks()` wrapper doesn't
expose `.bind_tools()` on the combined runnable anyway. A hand-parsed
ReAct-style protocol (TOOL_CALL: {...} / FINAL_ANSWER: ...) works
identically across every provider in the chain, and — usefully — the
assignment explicitly asks for "malformed or invented tool calls handled
without crashing", which a hand-rolled parser has to handle by construction
rather than being able to hide behind a provider's own validation.

Flow
────
  for round in range(MAX_ROUNDS):
      ask the model for exactly one TOOL_CALL or FINAL_ANSWER
      TOOL_CALL  → run the tool (never raises — errors become a string
                   result fed back to the model so it can recover),
                   yield a status event describing the call, append to
                   the scratchpad, loop
      FINAL_ANSWER → done
  if MAX_ROUNDS exhausted → force one last answer-only call

The final answer goes through the exact same quote_annotator pipeline as
the standard chat path — "verified quotes still apply to the final answer"
applies here identically, via the same QUOTE_MARK protocol generate() uses.
"""

import json
import logging
import re
from dataclasses import dataclass

from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langsmith import traceable

from app.services.langgraph_rag import invoke_with_retry, QUOTE_MARK
from app.services.embeddings import embed_text
from app.services.quote_annotator import DocRef as DocumentRef
from app.db.database import get_conn

logger = logging.getLogger("agent_research")

MAX_ROUNDS = 6


# ── Clause detection (heuristic, contract-style headings) ──────────────────

_HEADING_PATTERNS = [
    re.compile(r"^\s*\d+(\.\d+)*[\.\)]\s+\S.{2,100}$"),                               # "1. Definitions" / "2.1) Payment Terms"
    re.compile(r"^\s*(Section|Article|Clause)\s+\d+[A-Za-z]?\s*[:.\-]?\s*.{0,100}$", re.IGNORECASE),
    re.compile(r"^\s*[A-Z][A-Z \-/&',]{4,80}\s*$"),                                   # ALL CAPS short heading line
]


def detect_clauses(text: str) -> list[dict]:
    """Best-effort heading detection — not a real document-structure parser,
    just regex over likely contract heading shapes. Good enough to give the
    agent a navigable table of contents; not claimed to be exhaustive."""
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


# ── Tools ────────────────────────────────────────────────────────────────────

def _tool_list_clauses(doc: DocumentRef) -> str:
    clauses = detect_clauses(doc.text)
    if not clauses:
        return "No clearly-headed clauses/sections were detected in this document."
    return "\n".join(f"{i + 1}. {c['heading']}" for i, c in enumerate(clauses))


def _tool_get_section(number, doc: DocumentRef) -> str:
    try:
        number = int(number)
    except (TypeError, ValueError):
        return "ERROR: get_section requires an integer 'number' argument."
    clauses = detect_clauses(doc.text)
    if not clauses:
        return "ERROR: no clauses have been detected yet — call list_clauses() first."
    if number < 1 or number > len(clauses):
        return f"ERROR: section {number} doesn't exist. There are {len(clauses)} sections — call list_clauses() to see them."
    c = clauses[number - 1]
    section_text = doc.text[c["char_start"]:c["char_end"]].strip()
    return f"[Section {number}: {c['heading']}]\n{section_text[:4000]}"


def _tool_search_document(query, doc: DocumentRef) -> str:
    if not isinstance(query, str) or not query.strip():
        return "ERROR: search_document requires a non-empty string 'query' argument."
    try:
        query_embedding = embed_text([query])[0]
    except Exception as exc:
        return f"ERROR: search failed ({exc})."

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT content, score
                FROM hybrid_search(
                    query_text := %s,
                    query_embedding := %s::vector,
                    match_count := 5,
                    filter_document_ids := %s::uuid[],
                    rrf_k := 60
                )
                """,
                (query, str(query_embedding), [doc.document_id]),
            )
            rows = cur.fetchall()

    if not rows:
        return f"No passages matched '{query}' in this document."
    return "\n\n---\n\n".join(r[0] for r in rows)


def _execute_tool(name: str, args: dict, doc: DocumentRef) -> str:
    """Never raises — every failure mode (unknown tool, bad args, internal
    error) becomes a string result fed back to the model instead of
    crashing the request, per the spec's malformed-tool-call requirement."""
    try:
        if name == "search_document":
            return _tool_search_document(args.get("query"), doc)
        if name == "get_section":
            return _tool_get_section(args.get("number"), doc)
        if name == "list_clauses":
            return _tool_list_clauses(doc)
        return f"ERROR: unknown tool '{name}'. Available tools: search_document, get_section, list_clauses."
    except Exception as exc:
        logger.warning("[agent] Tool '%s' raised: %s", name, exc)
        return f"ERROR: tool '{name}' failed unexpectedly."


def _describe_call(name: str, args: dict) -> str:
    if name == "search_document":
        q = args.get("query", "")
        return f"Searching for {q}..." if q else "Searching the document..."
    if name == "get_section":
        return f"Reading section {args.get('number', '?')}..."
    if name == "list_clauses":
        return "Listing document clauses..."
    return f"Calling {name}..."


# ── Action parsing ───────────────────────────────────────────────────────────

@dataclass
class AgentAction:
    kind: str  # "tool_call" | "final" | "malformed"
    name: str | None = None
    args: dict | None = None
    text: str | None = None


_FINAL_RE = re.compile(r"FINAL_ANSWER:\s*(.*)", re.DOTALL)
_TOOL_PREFIX_RE = re.compile(r"TOOL_CALL:\s*")


def _extract_json_object(s: str, start: int) -> "str | None":
    """Scan forward from the first '{' at/after `start` and return the
    substring up to its MATCHING closing '}', tracking string literals so a
    brace inside a quoted value doesn't throw off the depth count.

    A non-greedy regex (`\\{.*?\\}`) breaks on any nested object — it stops
    at the first '}' it sees, which for {"name": ..., "arguments": {...}}
    is the inner object's close, not the outer one, producing unbalanced
    JSON on every tool call with a non-empty arguments dict. A greedy
    regex (`\\{.*\\}`) over-captures if there's any trailing text. Proper
    brace matching is the only approach that's correct for both.
    """
    open_idx = s.find("{", start)
    if open_idx == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(open_idx, len(s)):
        ch = s[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return s[open_idx:i + 1]
    return None


def parse_agent_action(raw: str) -> AgentAction:
    raw = raw.strip()

    final_match = _FINAL_RE.search(raw)
    tool_prefix_match = _TOOL_PREFIX_RE.search(raw)
    tool_json = _extract_json_object(raw, tool_prefix_match.end()) if tool_prefix_match else None

    # A model can emit both in one turn (ignoring the "exactly one" rule) —
    # prefer whichever comes first in the text as the intended action.
    if final_match and (not tool_prefix_match or final_match.start() <= tool_prefix_match.start()):
        return AgentAction(kind="final", text=final_match.group(1).strip())

    if tool_prefix_match:
        if tool_json is None:
            return AgentAction(kind="malformed")
        try:
            payload = json.loads(tool_json)
        except json.JSONDecodeError:
            return AgentAction(kind="malformed")
        name = payload.get("name") if isinstance(payload, dict) else None
        args = payload.get("arguments") if isinstance(payload, dict) else None
        if not isinstance(name, str) or not isinstance(args, dict):
            return AgentAction(kind="malformed")
        return AgentAction(kind="tool_call", name=name, args=args)

    if final_match:
        return AgentAction(kind="final", text=final_match.group(1).strip())

    return AgentAction(kind="malformed")


# ── Prompting ────────────────────────────────────────────────────────────────

_SYSTEM_TEMPLATE = """You are a document research agent investigating a single document to answer the user's question. You do NOT see the document's full text directly — you must use tools to look things up.

Available tools:
  - search_document(query: string) — semantic + keyword search over the document; returns the most relevant passages for the query.
  - get_section(number: integer) — returns the full text of clause/section N (1-based), as numbered by list_clauses().
  - list_clauses() — returns a numbered table of contents of the document's detected clauses/sections.

On every turn, respond with EXACTLY ONE of the following and nothing else:
  TOOL_CALL: {"name": "<tool name>", "arguments": {...}}
  FINAL_ANSWER: <your complete answer text>

You have at most {max_rounds} tool calls total ({calls_made} used so far) — use them purposefully. A reasonable approach: list_clauses() or a targeted search_document() for the relevant topic, then get_section() on anything promising. Don't call tools you don't need, and don't repeat an identical call.

Quoting rule: whenever your final answer states something the document directly supports, wrap the EXACT text of the supporting passage — copied character-for-character from a tool result, never paraphrased — like this: {quote_mark}exact quoted text{quote_mark}. Do not wrap text that isn't a verbatim quote.

Coverage rule: you have only seen what you explicitly looked up via tools, not the whole document. If you didn't look something up, or a search found nothing, say so in your final answer rather than claiming the document does or doesn't contain something you never actually checked.

Document: {filename}"""


def _build_messages(question: str, scratchpad: list[tuple[str, str]], doc: DocumentRef, calls_made: int) -> list[BaseMessage]:
    """Builds real LangChain message objects rather than going through
    ChatPromptTemplate.from_messages() — that treats '{'/'}' in message
    content as f-string template placeholders, which crashes (confirmed:
    "Invalid format specifier... Nested replacement fields are not
    allowed") on the very first scratchpad entry, since every TOOL_CALL /
    TOOL_RESULT is itself a JSON object full of braces. Raw message objects
    aren't template-interpolated at all, so arbitrary tool output is safe
    verbatim, with no escaping gymnastics needed anywhere in this module.
    """
    system_text = (
        _SYSTEM_TEMPLATE
        .replace("{max_rounds}", str(MAX_ROUNDS))
        .replace("{calls_made}", str(calls_made))
        .replace("{quote_mark}", QUOTE_MARK)
        .replace("{filename}", doc.filename)
    )
    messages: list[BaseMessage] = [SystemMessage(content=system_text), HumanMessage(content=f"Question: {question}")]
    for role, content in scratchpad:
        messages.append(AIMessage(content=content) if role == "assistant" else HumanMessage(content=content))
    return messages


# ── Public entry point ────────────────────────────────────────────────────────

@traceable(name="agent_research_loop", run_type="chain")
def run_agent_research(question: str, doc: DocumentRef, llm):
    """
    Generator mirroring run_rag_graph's shape: yields {"type": "status", ...}
    for every tool call (real progress, not a spinner) and finally
    {"type": "done", "answer": ...}.
    """
    parser = StrOutputParser()
    scratchpad: list[tuple[str, str]] = []
    calls_made = 0

    def _ask(messages: list[BaseMessage]) -> str:
        # RunnableLambda ignores the {} input invoke_with_retry passes and
        # always invokes the LLM with these specific messages — this is
        # what lets invoke_with_retry's existing retry/fallback machinery
        # be reused without going through a templated prompt.
        runnable = RunnableLambda(lambda _: llm.invoke(messages)) | parser
        return invoke_with_retry(runnable, {})

    for round_i in range(MAX_ROUNDS):
        messages = _build_messages(question, scratchpad, doc, calls_made)
        raw = _ask(messages)
        action = parse_agent_action(raw)

        if action.kind == "final":
            yield {"type": "done", "answer": action.text}
            return

        if action.kind == "malformed":
            logger.info("[agent] Malformed response (round %d): %r", round_i, raw[:200])
            scratchpad.append(("assistant", raw))
            scratchpad.append((
                "human",
                "ERROR: your response didn't match the required format. Respond with exactly one "
                "TOOL_CALL: {\"name\": ..., \"arguments\": {...}} or FINAL_ANSWER: ... line.",
            ))
            continue

        # tool_call
        calls_made += 1
        yield {"type": "status", "label": _describe_call(action.name, action.args)}
        result = _execute_tool(action.name, action.args, doc)
        scratchpad.append(("assistant", raw))
        scratchpad.append(("human", f"TOOL_RESULT: {result}"))

    # Round budget exhausted — force a final answer with whatever was gathered.
    yield {"type": "status", "label": "Reached the tool-call limit — finalizing with what was found..."}
    scratchpad.append((
        "human",
        "You've used all available tool calls. Respond now with FINAL_ANSWER: based on what you found.",
    ))
    messages = _build_messages(question, scratchpad, doc, calls_made)
    raw = _ask(messages)
    action = parse_agent_action(raw)
    if action.kind == "final":
        answer = action.text
    else:
        # The model ignored the forced-FINAL_ANSWER instruction (emitted
        # another TOOL_CALL, or something unparseable) — using `raw` here
        # would leak internal protocol text (literal "TOOL_CALL: {...}")
        # into the chat as if it were the answer. Degrade to an honest
        # message instead; whatever was actually found is still in the
        # transcript above and was already shown via the live status line.
        answer = (
            "I reached the tool-call limit while researching this before I could finish forming "
            "an answer. Try asking a more specific question, or break this into smaller questions."
        )
    yield {"type": "done", "answer": answer}
