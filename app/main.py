from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
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

app.include_router(router)
app.include_router(cache_router)
app.include_router(admin_router)