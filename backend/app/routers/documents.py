"""Document management endpoints: upload, list and delete."""
from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, Request, UploadFile

from app.core.embeddings import EmbeddingError
from app.schemas import DocumentInfo, UploadResponse

router = APIRouter(prefix="/documents", tags=["documents"])

ALLOWED = (".pdf", ".txt", ".md")


@router.post("/upload", response_model=UploadResponse)
async def upload(request: Request, files: list[UploadFile] = File(...)):
    engine = request.app.state.engine
    settings = engine.settings
    results: list[DocumentInfo] = []

    for f in files:
        name = f.filename or "upload"
        if not name.lower().endswith(ALLOWED):
            raise HTTPException(
                status_code=415,
                detail=f"Unsupported file type: {name}. Allowed: {', '.join(ALLOWED)}",
            )
        data = await f.read()
        if len(data) > settings.max_upload_mb * 1024 * 1024:
            raise HTTPException(
                status_code=413,
                detail=f"{name} exceeds the {settings.max_upload_mb} MB limit.",
            )
        try:
            info = engine.ingest_document(name, data)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except EmbeddingError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        results.append(DocumentInfo(**info))

    return UploadResponse(
        documents=results, total_chunks=sum(d.chunks for d in results)
    )


@router.get("", response_model=list[DocumentInfo])
async def list_documents(request: Request):
    engine = request.app.state.engine
    return [DocumentInfo(**d) for d in engine.store.documents()]


@router.delete("/{doc_id}")
async def delete_document(request: Request, doc_id: str):
    engine = request.app.state.engine
    removed = engine.store.delete(doc_id)
    if removed == 0:
        raise HTTPException(status_code=404, detail="Document not found.")
    return {"deleted": doc_id, "chunks_removed": removed}
