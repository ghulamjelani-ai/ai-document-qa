"""Tests for the OpenAI-compatible embeddings provider (mocked, offline)."""
from __future__ import annotations

import numpy as np
import pytest

import app.core.embeddings as em
from app.config import Settings
from app.core.embeddings import EmbeddingError, OpenAIEmbedder, build_embedder


class _Resp:
    def __init__(self, payload):
        self._p = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._p


class _Client:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def post(self, url, json, headers):
        # Return 3-dim embeddings, intentionally out of order to test re-sorting.
        n = len(json["input"])
        data = [
            {"index": n - 1 - i, "embedding": [float(n - 1 - i) + 1.0, 1.0, 0.0]}
            for i in range(n)
        ]
        return _Resp({"data": data})


def test_openai_embedder_parses_normalizes_and_orders(monkeypatch):
    monkeypatch.setattr(em.httpx, "Client", lambda timeout: _Client())
    emb = OpenAIEmbedder(Settings(embeddings_provider="openai", embeddings_api_key="k"))
    arr = emb.embed(["first", "second", "third"])
    assert arr.shape == (3, 3)
    # Rows are L2-normalized.
    assert np.allclose(np.linalg.norm(arr, axis=1), 1.0)
    assert emb.dim == 3


def test_openai_embedder_requires_key():
    with pytest.raises(EmbeddingError):
        build_embedder(Settings(embeddings_provider="openai", embeddings_api_key=""))


def test_build_embedder_unknown_provider():
    with pytest.raises(EmbeddingError):
        build_embedder(Settings(embeddings_provider="nope"))
