import logging
from fastapi import APIRouter, HTTPException
from langsmith import Client
from app.core.config import LANGCHAIN_API_KEY, LANGCHAIN_PROJECT

logger = logging.getLogger("admin")
router = APIRouter(prefix="/api/admin", tags=["admin"])

@router.get("/metrics")
def get_langsmith_metrics():
    """Fetch live observability metrics and recent traces from LangSmith."""
    project_name = LANGCHAIN_PROJECT
    api_key = LANGCHAIN_API_KEY

    if not api_key or api_key == "your_langsmith_api_key_here":
        raise HTTPException(status_code=503, detail="LangSmith API key not configured in .env")
        
    try:
        client = Client(api_key=api_key)
        
        # Fetch the most recent 50 root runs (traces) for this project
        runs = list(client.list_runs(
            project_name=project_name,
            is_root=True,  # Only get the top-level trace, not every inner step
            limit=50
        ))
        
        total_runs = len(runs)
        success_runs = sum(1 for r in runs if not r.error)
        success_rate = (success_runs / total_runs * 100) if total_runs > 0 else 100
        
        latencies = []
        total_tokens = 0
        recent_activity = []
        
        for r in runs:
            # Latency
            latency_ms = 0
            if r.end_time and r.start_time:
                latency_ms = (r.end_time - r.start_time).total_seconds() * 1000
                latencies.append(latency_ms)
                
            # Tokens (LangSmith puts these in the run outputs or sometimes root metrics)
            tokens = 0
            if hasattr(r, 'total_tokens') and r.total_tokens:
                tokens = r.total_tokens
            elif hasattr(r, 'prompt_tokens') and r.prompt_tokens:
                tokens = r.prompt_tokens + getattr(r, 'completion_tokens', 0)
                
            total_tokens += tokens
            
            # Add to recent activity list (top 15)
            if len(recent_activity) < 15:
                recent_activity.append({
                    "id": str(r.id),
                    "name": r.name,
                    "status": "error" if r.error else "success",
                    "latency_ms": round(latency_ms),
                    "tokens": tokens,
                    "start_time": r.start_time.isoformat() if r.start_time else None
                })
                
        avg_latency = (sum(latencies) / len(latencies)) if latencies else 0
        
        return {
            "metrics": {
                "total_runs": total_runs,
                "success_rate": round(success_rate, 1),
                "avg_latency_ms": round(avg_latency),
                "total_tokens": total_tokens
            },
            "recent_activity": recent_activity
        }
        
    except Exception as e:
        logger.error(f"[LangSmith] Error fetching metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch metrics from LangSmith: {str(e)}")
