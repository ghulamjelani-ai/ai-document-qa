"""Shared test fixtures.

Tests run fully offline: the embedder and LLM are replaced with deterministic
fakes, so no Ollama server or network is required.
"""
from __future__ import annotations

import hashlib
import os
import tempfile

import numpy as np
import pytest

# Point the app at a throwaway data directory BEFORE importing it.
_TMP = tempfile.mkdtemp(prefix="aidocqa-test-")
os.environ["DATA_DIR"] = _TMP

from fastapi.testclient import TestClient  # noqa: E402

from app.config import get_settings  # noqa: E402

get_settings.cache_clear()

DIM = 24


class FakeEmbedder:
    """Deterministic, normalized embeddings derived from a hash of the text."""

    dim = DIM

    def embed(self, texts: list[str]) -> np.ndarray:
        rows = []
        for t in texts:
            seed = int(hashlib.sha256(t.lower().encode()).hexdigest(), 16) % (2**32)
            rng = np.random.default_rng(seed)
            # Blend a hash vector with simple word-presence signal so that similar
            # text lands near each other (good enough for retrieval assertions).
            v = rng.standard_normal(DIM).astype("float32")
            for i, word in enumerate(sorted(set(t.lower().split()))[:DIM]):
                v[i % DIM] += (len(word) % 5) + 1
            rows.append(v)
        arr = np.asarray(rows, dtype="float32")
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return arr / norms


class FakeLLM:
    name = "fake-model"

    def complete(self, system: str, user: str) -> str:
        return "Based on the context: ANSWERED."


@pytest.fixture
def client():
    from app.main import app

    with TestClient(app) as c:
        app.state.engine._embedder = FakeEmbedder()
        app.state.engine._llm = FakeLLM()
        app.state.engine.store.clear()
        yield c
