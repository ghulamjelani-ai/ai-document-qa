"""Read documents and split them into overlapping text chunks.

Supported formats: PDF (.pdf), plain text (.txt), Markdown (.md).
Each chunk keeps the page number it came from (PDFs) so answers can cite it.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class Chunk:
    text: str
    chunk_index: int
    page: int | None


def _clean(text: str) -> str:
    # Collapse runs of whitespace but keep paragraph breaks readable.
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_pages(filename: str, data: bytes) -> list[tuple[int | None, str]]:
    """Return a list of (page_number, text). Non-PDF files are one 'page' (None)."""
    lower = filename.lower()
    if lower.endswith(".pdf"):
        from pypdf import PdfReader
        from io import BytesIO

        reader = PdfReader(BytesIO(data))
        pages: list[tuple[int | None, str]] = []
        for i, page in enumerate(reader.pages, start=1):
            text = _clean(page.extract_text() or "")
            if text:
                pages.append((i, text))
        return pages
    if lower.endswith((".txt", ".md")):
        return [(None, _clean(data.decode("utf-8", errors="replace")))]
    raise ValueError(f"Unsupported file type: {filename}")


def chunk_text(
    text: str, chunk_size: int, overlap: int, page: int | None, start_index: int
) -> list[Chunk]:
    """Split one page/body of text into overlapping chunks at sentence-ish boundaries."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    overlap = max(0, min(overlap, chunk_size - 1))

    chunks: list[Chunk] = []
    idx = start_index
    pos = 0
    length = len(text)
    while pos < length:
        end = min(pos + chunk_size, length)
        # Try to end on a sentence/paragraph boundary for cleaner context.
        if end < length:
            window = text[pos:end]
            for sep in ("\n\n", ". ", ".\n", "? ", "! ", "\n"):
                cut = window.rfind(sep)
                if cut > chunk_size * 0.5:
                    end = pos + cut + len(sep)
                    break
        piece = text[pos:end].strip()
        if piece:
            chunks.append(Chunk(text=piece, chunk_index=idx, page=page))
            idx += 1
        if end >= length:
            break
        # Advance, leaving `overlap` characters of context for the next chunk.
        next_pos = end - overlap if overlap else end
        pos = next_pos if next_pos > pos else end  # guarantee forward progress
    return chunks


def ingest(
    filename: str, data: bytes, chunk_size: int, overlap: int
) -> list[Chunk]:
    """Full pipeline: bytes -> cleaned pages -> overlapping chunks."""
    pages = extract_pages(filename, data)
    chunks: list[Chunk] = []
    for page, text in pages:
        chunks.extend(
            chunk_text(text, chunk_size, overlap, page, start_index=len(chunks))
        )
    return chunks
