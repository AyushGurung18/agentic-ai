import os
from dotenv import load_dotenv

load_dotenv() # Load variables from .env

CHUNK_SIZE    = 500
CHUNK_OVERLAP = 100
TOP_K         = 3

# ── Self-hosted inference (vLLM, OpenAI-compatible server) ───────────────────
# Used as the no-API-key-needed fallback when no cloud provider key is set.
# vLLM's PagedAttention KV-cache manager is what makes this serving path
# viable for concurrent requests without per-request GPU memory blowup.
VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8001/v1")
VLLM_MODEL    = os.environ.get("VLLM_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct")
VLLM_API_KEY  = os.environ.get("VLLM_API_KEY", "not-needed")

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY", "")
OPENROUTER_API_KEY = os.environ.get("OPENROUTERAPIKEY", "")

# Database
# Strip SQLAlchemy driver prefix if present for compatibility
DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://ayush:password@localhost:5432/octo",
)
POSTGRES_URL = DATABASE_URL.replace("postgresql+psycopg://", "postgresql://")

# Debugging: Print connection host/port (safe version)
if "db.lmpnnnfbfyclfwqwbbgd.supabase.co" in POSTGRES_URL:
    print(f"📡 Connecting to Supabase Project: lmpnnnfbfyclfwqwbbgd")
    if ":6543" in POSTGRES_URL:
        print("✅ Using Transaction Pooler (Port 6543)")
    else:
        print("⚠️ Warning: Using direct connection (Port 5432). This may fail on IPv4-only networks.")

# Default dev user — used when no JWT token is present (local dev only).
DEV_USER_ID    = os.environ.get("DEV_USER_ID",    "00000000-0000-0000-0000-000000000001")
DEV_USER_EMAIL = os.environ.get("DEV_USER_EMAIL",  "dev@local")

# Supabase Auth — JWT secret from Dashboard → Settings → API → JWT Settings
SUPABASE_JWT_SECRET = os.environ.get("SUPABASE_JWT_SECRET", "")
SUPABASE_URL        = os.environ.get("SUPABASE_URL", "")
SUPABASE_ANON_KEY   = os.environ.get("SUPABASE_ANON_KEY", "")

# ── LangSmith (Observability) ────────────────────────────────────────────────
# LangSmith renamed its env vars from LANGCHAIN_* to LANGSMITH_* — check the
# new names first, fall back to the legacy ones for compatibility.
LANGCHAIN_TRACING_V2 = os.getenv("LANGSMITH_TRACING", os.getenv("LANGCHAIN_TRACING_V2", "false")).lower() == "true"
LANGCHAIN_API_KEY = os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY")
LANGCHAIN_PROJECT = os.getenv("LANGSMITH_PROJECT") or os.getenv("LANGCHAIN_PROJECT", "Thotqen-Prod-RAG")

# ── Celery / RabbitMQ (CloudAMQP) ─────────────────────────────────────────────
# CloudAMQP provides this URL from the dashboard.
# Format: amqps://user:pass@kangaroo.rmq.cloudamqp.com/vhost
CLOUDAMQP_URL = os.environ.get("CLOUDAMQP_URL", "")

# ── CRAG Web Search (optional Tavily upgrade) ─────────────────────────────────
# Leave unset to use DuckDuckGo (free, no key needed).
# Set to enable Tavily for higher-quality web search results.
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY", "")