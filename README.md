---
title: Thotqen Backend
emoji: 🚀
colorFrom: blue
colorTo: green
sdk: docker
---

# 🚀 Thotqen: Production-Grade Conversational RAG Engine

A highly optimized, production-ready Retrieval-Augmented Generation (RAG) system built to solve the real-world trade-offs of latency, cost, and retrieval accuracy. Engineered with a hybrid storage layer, stateful agentic flows, and multi-model routing.

---

## ⚡ Architectural Highlights & Production-Grade Decoupling

Unlike basic "tutorial-grade" RAG systems, **Thotqen** implements advanced optimization patterns to handle production scale:

### 1. Stateful Self-RAG & CRAG Graph ([langgraph_rag.py](file:///home/ayush/.dev/bbygrl/thotqen/app/services/langgraph_rag.py))
Built on **LangGraph**, the pipeline manages a complex routing topology that incorporates:
*   **Input Guardrails:** Early classification to block prompt injection and toxic queries.
*   **HyDE (Hypothetical Document Embeddings):** Generates hypothetical answers to bridge the semantic gap between questions and document chunks.
*   **Corrective RAG (CRAG) Fallback:** Integrates live web search (Tavily/DuckDuckGo) when the local knowledge base yields insufficient document scores.
*   **Self-RAG Loops:** Evaluates generated answers for hallucinations (groundedness) and query alignment (usefulness), triggering automatic query-rewrites if thresholds aren't met.

### 2. Hierarchical (Parent-Child) Chunking ([chunking.py](file:///home/ayush/.dev/bbygrl/thotqen/app/services/chunking.py))
*   **The Problem:** Large chunks dilute semantic vector representation. Tiny chunks lack the context necessary for an LLM to generate high-quality answers.
*   **The Solution:** Thotqen indexes **300-character child chunks** for high-precision vector search, but resolves them back to **1500-character parent chunks** in SQL before feeding them to the generation LLM.

### 3. PostgreSQL Hybrid Search with Reciprocal Rank Fusion (RRF) ([20260713_advanced_rag_schema.sql](file:///home/ayush/.dev/bbygrl/thotqen/supabase/migrations/20260713_advanced_rag_schema.sql))
Uses a custom database-level function `hybrid_search()` to run:
*   **Dense Retrieval:** HNSW vector similarity search on `pgvector`.
*   **Sparse Retrieval:** Full-text keyword matching using PostgreSQL `tsvector` with a GIN index.
*   **Fusion:** Fuses ranks using the **RRF formula** ($Score = \sum \frac{1}{60 + rank}$), catching exact terms (codes, IDs) and semantic meaning without needing complex score-normalization.

### 4. Intent Routing ([rag.py](file:///home/ayush/.dev/bbygrl/thotqen/app/services/rag.py))
*   Routes incoming queries through a cheap classifier (`llama-3.1-8b-instant`).
*   **Cheap/Factual Queries** are handled by high-throughput models like `gemini-1.5-flash`.
*   **Complex/Reasoning Queries** are routed to `meta/llama-3.1-70b-instruct` (Nvidia API) or `gemini-1.5-pro`. This reduces average API costs by up to 80%.

### 5. Local Cross-Encoder Reranking
*   Employs a local `cross-encoder/ms-marco-MiniLM-L-6-v2` model to evaluate the retrieved document list.
*   Trims down the document pool to the top 5 most highly correlated contexts, minimizing the LLM context window and preventing "lost in the middle" retrieval degradation.

### 6. Asynchronous Background Ingestion
*   Decoupled ingestion pipeline using **Celery** + Redis/CloudAMQP.
*   Large file processing, parsing, hierarchical chunking, and embedding generation are run asynchronously, reporting real-time progress to the client via Server-Sent Events (SSE).

### 7. Evaluation Suite ([evaluate_sessions.py](file:///home/ayush/.dev/bbygrl/thotqen/scripts/evaluate_sessions.py))
*   Includes offline telemetry validation using the **Ragas** framework.
*   Evaluates production Q&A outputs on metrics like `faithfulness` (hallucination detection) and `answer_relevance`.

---

## 🛠️ Stack & Infrastructure
*   **Framework:** FastAPI (Python)
*   **Orchestration & Workflow:** LangGraph, Celery
*   **Vector Database:** Supabase PostgreSQL with `pgvector` & HNSW indexing
*   **Embeddings:** Local SentenceTransformers (`all-MiniLM-L6-v2` / 384-dim)
*   **LLMs:** Gemini Pro/Flash (Google), Llama 3.x (Groq & Nvidia AI Endpoints)
*   **Hosting:** Dockerized deployment optimized for Hugging Face Spaces

---

## 📝 Environment Setup
Ensure the following keys are set in your `.env` file or hosting provider's variables:
```bash
DATABASE_URL=postgresql://...
SECRET_KEY=your_auth_secret
GROQ_API_KEY=gsk_...
GEMINI_API_KEY=AIzaSy...
NVIDIA_API_KEY=nvapi-...
TAVILY_API_KEY=tvly-...
```