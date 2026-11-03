"""Question answering endpoint."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from app.core.embeddings import EmbeddingError
from app.core.llm import LLMError
from app.schemas import AskRequest, AskResponse, Source

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("/ask", response_model=AskResponse)
async def ask(request: Request, body: AskRequest):
    engine = request.app.state.engine
    if engine.store.stats()["chunks"] == 0:
        raise HTTPException(
            status_code=409, detail="No documents have been uploaded yet."
        )
    try:
        result = engine.answer(body.question, body.top_k, source=body.source)
    except EmbeddingError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except LLMError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    sources = [
        Source(
            doc_id=h.doc_id,
            filename=h.filename,
            chunk_index=h.chunk_index,
            page=h.page,
            score=round(h.score, 4),
            snippet=(h.text[:300] + "…") if len(h.text) > 300 else h.text,
        )
        for h in result.hits
    ]
    return AskResponse(
        answer=result.answer,
        sources=sources,
        used_context=result.used_context,
        model=result.model,
        retrieval_mode=result.retrieval_mode,
    )
