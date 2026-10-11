#!/usr/bin/env python3
"""Render updated architecture diagrams via mermaid.ink."""

import base64
import json
import urllib.request
import os

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "docs")


def render_mermaid(code: str, out_path: str, theme: str = "neutral"):
    obj = {"code": code, "mermaid": {"theme": theme}}
    b64 = base64.urlsafe_b64encode(json.dumps(obj).encode()).decode()
    url = f"https://mermaid.ink/img/{b64}?bgColor=FFFFFF&width=1600"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = resp.read()
    with open(out_path, "wb") as f:
        f.write(data)
    print(f"  ✓ Saved {out_path} ({len(data)//1024} KB)")


# ── Diagram 1: System Architecture Overview ────────────────────────────────────
ARCH_OVERVIEW = """
flowchart LR
    subgraph Frontend["Frontend · bbygrl (Next.js)"]
        direction TB
        UI["Contract Workspace UI\\nVoice Input · RTL Arabic Detection\\nExport (.md) · Supabase Auth"]
        DocViewer["Synced Document Viewer\\nPDF / DOCX + Citation Highlighting"]
        DiffUI["Document Version Comparator\\nParagraph Diffs + Significance Badges"]
    end

    subgraph Edge["Cloudflare Edge Gateway"]
        direction TB
        Proxy["Reverse Proxy\\nCORS · timeout + retry"]
        Cache{"Semantic Cache\\nKV lookup"}
        DocServe["R2 File Serve\\n/edge/docs/*"]
    end

    subgraph API["Backend · thotqen (FastAPI on HF Spaces)"]
        direction TB
        UploadEP["POST /upload\\nPDF & DOCX dispatch"]
        ChatEP["POST /chat\\nstandard | agentic mode"]
        CompareEP["POST /documents/compare"]
        AdminEP["GET /api/admin/metrics\\nLangSmith live stats"]
    end

    subgraph Ingestion["Async Ingestion (Celery + CloudAMQP)"]
        direction TB
        PyMuPDF["PyMuPDF / python-docx\\nText + Page Offset Map"]
        Chunker["Hierarchical Chunker\\nParent ~1500t → Child ~300t"]
        Embedder["Local MiniLM-L6-v2\\n384-dim vectors"]
    end

    subgraph Store["Storage"]
        direction TB
        PG[("Postgres + pgvector\\nHNSW + BM25 tsvector\\n+ hybrid_search()")]
        R2[("Cloudflare R2\\nRaw PDF / DOCX")]
        KV[("Cloudflare KV\\nSemantic cache")]
    end

    subgraph Pipeline["Query Execution"]
        direction TB
        Intent{"route_intent\\nfast LLM call"}
        CRAG["CRAG Pipeline\\nHyDE → Hybrid RRF → BGE Rerank\\n→ Grade → Web fallback → Generate"]
        Agent["Agentic Research Engine\\nReAct Loop · 6-round max\\nsearch / get_section / list_clauses"]
        Compare["Comparison Engine\\nJaccard preflight + difflib\\n+ LLM significance grading"]
    end

    subgraph Safety["Verification & Tracing"]
        direction TB
        Verif["Quote Verifier\\nwhitespace-tolerant regex\\nchar + page span location"]
        LS["LangSmith\\nfull trace tree per request"]
        RAGAS["RAGAS eval harness\\nfaithfulness · relevancy · precision · recall"]
    end

    Frontend -- "REST + SSE" --> Edge
    Edge --> Proxy
    Proxy --> Cache
    Cache -- hit --> KV
    Cache -- miss --> API
    DocServe --> R2
    API --> Ingestion
    API --> Pipeline
    CompareEP --> Compare
    Ingestion --> PyMuPDF --> Chunker --> Embedder --> PG
    UploadEP --> R2
    Intent --> CRAG
    Intent --> Agent
    CRAG --> PG
    Agent --> PG
    CRAG --> Verif
    Agent --> Verif
    Verif --> Frontend
    Pipeline --> LS --> RAGAS
"""

# ── Diagram 2: Query Pipeline Detail ────────────────────────────────────────────
PIPELINE_DETAIL = """
flowchart TD
    START(["User Question\\nVoice Speech or Text + attached document(s)"])
    INTENT{"route_intent\\nfast LLM call:\\ncheap or complex?\\nagentic mode on?"}

    START --> INTENT

    INTENT -- "standard · complex" --> CHAINX["LLM Chain — complex\\nGroq-2 → Gemini-2 → OpenRouter ×2\\n→ Groq-1 → Gemini-1 → vLLM\\n(18s hard cutoff each)"]
    INTENT -- "standard · cheap" --> CHAINC["LLM Chain — cheap\\nGroq-2 → Gemini-2 → OpenRouter ×2\\n→ Groq-1 → Gemini-1 → vLLM\\n(18s hard cutoff each)"]
    INTENT -- "agentic mode" --> AGENT

    CHAINX --> IG
    CHAINC --> IG

    IG["input_guardrail\\nLLM safety check"]
    IG -- flagged --> BLOCKED(["Request blocked"])
    IG -- safe --> HYDE["hyde_generator\\nHypothetical answer for recall boost"]

    HYDE --> RET["hybrid_search()\\nHNSW dense + BM25 sparse → RRF fusion\\nChild chunks → Parent context returned"]
    RET --> RERANK["BGE cross-encoder reranker\\nkeep top-5"]
    RERANK --> GRADE{"grade_documents\\n1 batched LLM call grades ALL docs\\n≥50% relevant?"}
    GRADE -- "yes" --> GEN
    GRADE -- "no, web unused" --> WEB["web_search\\nTavily (DuckDuckGo fallback)"]
    GRADE -- "no, web done" --> RW["rewrite_query\\nLLM rephrases"]
    WEB --> GEN
    RW -- "loop back (max 2 rounds)" --> RET

    GEN["generate\\nAnswer with QUOTE_MARK protocol\\n+ chat history context"]

    GEN --> VERIFY

    subgraph AGENT["Agentic Research Loop · ReAct protocol"]
        direction TB
        SYS["System prompt: tool spec + document text"]
        ROUND{"Round ≤ 6?"}
        LLM_CALL["LLM call:\\nTOOL_CALL or FINAL_ANSWER?"]
        PARSE{"Parse response"}
        TOOL_RUN["Execute tool\\n(never raises — errors → model feedback)"]
        SCRATCHPAD["Append tool result to scratchpad"]
        FORCE["Force final answer\\n(max rounds hit)"]
        SYS --> ROUND
        ROUND -- yes --> LLM_CALL --> PARSE
        PARSE -- "TOOL_CALL" --> TOOL_RUN --> SCRATCHPAD --> ROUND
        PARSE -- "FINAL_ANSWER" --> VERIFY
        ROUND -- "no (>6)" --> FORCE --> VERIFY
    end

    VERIFY["quote_verifier.py\\nWhitespace-tolerant regex per quote\\nChar offset + page span location\\nUnverified quotes flagged, not hidden"]
    VERIFY --> STREAM["SSE Stream & Export\\nAnnotated answer + verified quote spans\\n+ tool call status events\\n+ Markdown export & audio read-aloud"]
"""

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Rendering architecture_overview.png ...")
    render_mermaid(ARCH_OVERVIEW, os.path.join(OUTPUT_DIR, "architecture_overview.png"))

    print("Rendering pipeline_detail.png ...")
    render_mermaid(PIPELINE_DETAIL, os.path.join(OUTPUT_DIR, "pipeline_detail.png"))

    print("Done!")
