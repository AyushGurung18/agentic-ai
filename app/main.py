from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.routes import router
from app.cache_routes import router as cache_router
from app.admin_routes import router as admin_router
from app.db.database import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Run schema bootstrap on startup."""
    init_db()
    yield


app = FastAPI(
    title="AI Knowledge Copilot",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Return a single string `detail` so the UI can show why a 422 happened."""
    errors = exc.errors()
    missing_file = any(
        (err.get("loc") or ())[-1] == "file"
        for err in errors
    )
    if missing_file and "/upload" in request.url.path:
        return JSONResponse(
            status_code=422,
            content={"detail": "No file was received. Attach a PDF or DOCX and try again."},
        )
    messages = []
    for err in errors:
        loc = " → ".join(str(part) for part in err.get("loc", []) if part != "body")
        msg = err.get("msg") or "Invalid request"
        messages.append(f"{loc}: {msg}" if loc else msg)
    return JSONResponse(
        status_code=422,
        content={"detail": "; ".join(messages) or "The request was invalid."},
    )


app.include_router(router)
app.include_router(cache_router)
app.include_router(admin_router)