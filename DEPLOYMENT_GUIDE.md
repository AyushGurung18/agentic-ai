# 📦 Deployment Guide – ThotQen

## 1️⃣ Required Secrets & Environment Variables
| Variable | Where to set | Description |
|----------|--------------|-------------|
| **INTERNAL_API_SECRET** | `.env` (backend) **and** Cloudflare Worker KV binding (as a secret) | Shared secret that guards the internal FastAPI endpoints. Must be identical on both sides. |
| **CF_ACCOUNT_ID** | Cloudflare Worker **Variables** | Your Cloudflare account identifier. |
| **CF_KV_NAMESPACE_ID** | Cloudflare Worker **Variables** | Namespace ID for the `CHAT_CACHE` KV store. |
| **CF_API_TOKEN** | Cloudflare Worker **Variables** | API token with **Edit KV** permission (needs `Account Resources → Workers KV → Edit`). |
| **FASTAPI_ORIGIN** | Cloudflare Worker **Variables** | Base URL of the FastAPI service (e.g. `http://localhost:8000` for local testing or the public URL when deployed). |
| **DATABASE_URL**, **SUPABASE_URL**, **SUPABASE_SERVICE_ROLE_KEY**, **SUPABASE_JWT_SECRET** | `.env` (backend) | Supabase connection and auth credentials (already present). |
| **GROQ_API_KEY**, **R2_ACCESS_KEY_ID**, **R2_SECRET_ACCESS_KEY**, **R2_JURISDICTION_SPECIFIC_ENDPOINT** | `.env` (backend) | Model and R2 bucket credentials (already present). |

> **Tip** – In Hugging Face Spaces, add all these keys under **Settings → Secrets**. They become environment variables automatically.

## 2️⃣ Cloudflare Worker – How It Works
1. **GET `/api/chats/list/:userId`** –
   - Checks `CHAT_CACHE` KV for key `user_chats:<userId>`.
   - Logs
     ```js
     console.log('[Cache Hit] Serving from Edge KV')
     ```
   - If missing, proxies to FastAPI and logs `[Cache Miss] Proxying to FastAPI`.
2. **POST `/api/chat/stream`** –
   - Extracts `message` from the request body.
   - Calls `@cf/baai/bge-small-en-v1.5` to get a 384‑dim vector.
   - Calls FastAPI internal endpoint `POST /api/internal/vector-cache-check` **with** header `X-Internal-Secret: <INTERNAL_API_SECRET>`.
   - If the response reports a hit with `similarity > 0.96`, the worker returns the cached answer directly (log `[Cache Hit] Served from PostgreSQL semantic cache`).
   - Otherwise it proxies the original request to FastAPI.
3. **Cache Invalidation** – Whenever the backend creates, updates, or deletes a user’s chat session it calls `POST /api/internal/invalidate-user-chat/{userId}` (again with the secret header). The worker removes the corresponding KV entry.

All logs appear in the **Cloudflare Workers Dashboard → Logs** – you can monitor cache behaviour live.

## 3️⃣ FastAPI – Internal Endpoints (Protected)
- **Header Validation** – Every internal route depends on `validate_internal_secret`, which reads `INTERNAL_API_SECRET` from the environment and rejects mismatched requests with **401 Unauthorized**.
- **Endpoints** (mounted under `/api/internal`):
  - `POST /vector-cache-check` – Calls the PL/pgSQL function `match_semantic_cache` and returns `{hit, cached_answer?, similarity?}`.
  - `POST /cache-write` – Persists a new Q&A + embedding into `semantic_responses_cache`.
  - `POST /invalidate-user-chat/{user_id}` – Sends a DELETE request to Cloudflare KV via the official API.
- **Logging** – Uses a dedicated logger named `cache`. Example messages:
  ```
  [2026-05-28 12:34:56,789] INFO cache: [Cache Hit] Served from PostgreSQL semantic cache (similarity=0.9723)
  [2026-05-28 12:35:12,123] INFO cache: [Cache Miss] No semantic match in PostgreSQL cache
  ```

## 4️⃣ Building & Deploying to Hugging Face Spaces
1. **Commit the new `Dockerfile`** (UID 1000, non‑root, port 7860). The repository already contains the production‑ready file.
2. **Push to a GitHub/GitLab repo** that is linked to your HF Space, or simply clone the repo inside the Space UI and let HF build it.
3. **Add Secrets** – In the Space UI go to **Settings → Secrets** and paste every key from the table above (including `INTERNAL_API_SECRET`).
4. **Build** – HF automatically runs:
   ```bash
   docker build -t space .
   ```
5. **Run** – The container starts with:
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 7860
   ```
   – The health endpoint `GET /health` should return `{"status":"ok"}`.
6. **Verify** – Open the Space URL and test:
   * `GET /api/chats/list/<your‑id>` should return a cached empty list on first call, then a cached response on subsequent calls.
   * `POST /api/chat/stream` with a repeated question should hit the PostgreSQL cache (check the Worker logs for `[Cache Hit] Served from PostgreSQL semantic cache`).

## 5️⃣ Local Development Workflow
1. **Start FastAPI**:
   ```bash
   uvicorn app.main:app --reload
   ```
2. **Start the Worker** (requires `wrangler`):
   ```bash
   wrangler dev --env dev
   ```
   Ensure your local `.env` contains **all** secrets (including `INTERNAL_API_SECRET`).
3. **Test the flow** using `curl` or a REST client:
   ```bash
   # First request – should miss KV and proxy
   curl -X GET http://localhost:8787/api/chats/list/123

   # Second request – should hit KV
   curl -X GET http://localhost:8787/api/chats/list/123
   ```
   Look at the console output from `wrangler dev` for the cache‑hit/miss logs.

---
**Enjoy your dual‑layer caching system and smooth migration to Hugging Face Spaces!**
