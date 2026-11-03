"""Tests for text extraction and chunking."""
from __future__ import annotations

import pytest

from app.core.ingest import chunk_text, extract_pages, ingest


def test_chunking_respects_size_and_overlap():
    text = "word " * 500  # 2500 chars
    chunks = chunk_text(text, chunk_size=300, overlap=50, page=None, start_index=0)
    assert len(chunks) > 1
    assert all(len(c.text) <= 300 for c in chunks)
    # indices are sequential
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_chunking_makes_progress_and_covers_text():
    text = "A. " * 1000
    chunks = chunk_text(text, chunk_size=200, overlap=40, page=None, start_index=0)
    joined = " ".join(c.text for c in chunks)
    assert "A." in joined
    assert len(chunks) >= 5


def test_small_text_single_chunk():
    chunks = chunk_text("hello world", chunk_size=900, overlap=100, page=3, start_index=0)
    assert len(chunks) == 1
    assert chunks[0].page == 3


def test_extract_txt():
    pages = extract_pages("notes.txt", b"Line one.\nLine two.")
    assert pages == [(None, "Line one.\nLine two.")]


def test_unsupported_type_raises():
    with pytest.raises(ValueError):
        extract_pages("image.png", b"\x89PNG")


def test_ingest_txt_end_to_end():
    chunks = ingest("doc.txt", b"Some text here. " * 100, chunk_size=200, overlap=30)
    assert len(chunks) >= 1
    assert all(c.text for c in chunks)
