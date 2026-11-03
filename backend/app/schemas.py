"""Pydantic request/response models for the API."""
from __future__ import annotations

from pydantic import BaseModel, Field


class DocumentInfo(BaseModel):
    doc_id: str
    filename: str
    chunks: int
    characters: int


class UploadResponse(BaseModel):
    documents: list[DocumentInfo]
    total_chunks: int


class Source(BaseModel):
    doc_id: str
    filename: str
    chunk_index: int
    page: int | None = None
    score: float
    snippet: str


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=20)
    source: str | None = Field(
        default=None, description="Optional filename filter: search only this document."
    )


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]
    used_context: bool
    model: str
    retrieval_mode: str


class HealthResponse(BaseModel):
    status: str
    documents: int
    chunks: int


class ErrorBody(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    error: ErrorBody
