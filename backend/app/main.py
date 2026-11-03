"""FastAPI application: middleware, error handling and routes."""
from __future__ import annotations

import logging
import uuid
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.core.rag import RagEngine
from app.routers import chat, documents
from app.schemas import HealthResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("ai-document-qa")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.settings = settings
    app.state.engine = RagEngine(settings)
    logger.info(
        "Started: embeddings=%s llm=%s", settings.embeddings_provider, settings.llm_provider
    )
    yield


app = FastAPI(
    title="AI Document Q&A",
    description="Upload documents and ask grounded questions with cited sources (RAG).",
    version="1.0.0",
    lifespan=lifespan,
)

settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_url, "http://localhost:8501"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request_id = uuid.uuid4().hex[:12]
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


@app.exception_handler(HTTPException)
async def http_exc_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": {"code": f"http_{exc.status_code}", "message": exc.detail}},
        headers={"X-Request-ID": getattr(request.state, "request_id", "")},
    )


@app.exception_handler(Exception)
async def unhandled_exc_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error: %s", exc)
    return JSONResponse(
        status_code=500,
        content={"error": {"code": "internal_error", "message": "Something went wrong."}},
        headers={"X-Request-ID": getattr(request.state, "request_id", "")},
    )


app.include_router(documents.router)
app.include_router(chat.router)


@app.get("/", tags=["health"])
async def root():
    return {
        "name": "AI Document Q&A",
        "version": "1.0.0",
        "docs": "/docs",
    }


@app.get("/health", response_model=HealthResponse, tags=["health"])
async def health(request: Request):
    stats = request.app.state.engine.store.stats()
    return HealthResponse(status="ok", documents=stats["documents"], chunks=stats["chunks"])


@app.get("/ready", tags=["health"])
async def ready(request: Request):
    """Reports whether the embedding/LLM backend (Ollama) is reachable."""
    s = request.app.state.settings
    if s.embeddings_provider == "ollama" or s.llm_provider == "ollama":
        try:
            with httpx.Client(timeout=5) as client:
                client.get(s.ollama_base_url.rstrip("/") + "/api/tags").raise_for_status()
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=503, detail=f"Ollama not reachable: {exc}")
    return {"ready": True}
