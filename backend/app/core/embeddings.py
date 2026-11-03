"""Turn text into vectors. Three interchangeable providers.

- "ollama": calls a local Ollama server (default, no API key, free).
- "sentence_transformers": runs a local model in-process (offline, good for a
  showcase of the sentence-transformers stack). Imported lazily so the heavy
  dependency is only needed when this provider is actually selected.
- "openai": any OpenAI-compatible /embeddings API (base URL configurable), so the
  same code works with OpenAI, Jina, Google Gemini's OpenAI endpoint, Voyage, etc.
  This is what the hosted (cloud) deployment uses, since Ollama is local-only.
"""
from __future__ import annotations

import httpx
import numpy as np

from app.config import Settings


class EmbeddingError(RuntimeError):
    pass


class Embedder:
    """Base interface."""

    dim: int

    def embed(self, texts: list[str]) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError


class OllamaEmbedder(Embedder):
    def __init__(self, settings: Settings):
        self._url = settings.ollama_base_url.rstrip("/") + "/api/embeddings"
        self._model = settings.ollama_embedding_model
        self._timeout = settings.request_timeout
        self.dim = 0  # discovered on first call

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors: list[list[float]] = []
        try:
            with httpx.Client(timeout=self._timeout) as client:
                for text in texts:
                    resp = client.post(
                        self._url, json={"model": self._model, "prompt": text}
                    )
                    resp.raise_for_status()
                    vectors.append(resp.json()["embedding"])
        except httpx.HTTPError as exc:
            raise EmbeddingError(
                f"Ollama embedding request failed ({self._model}): {exc}"
            ) from exc
        arr = np.asarray(vectors, dtype="float32")
        self.dim = arr.shape[1] if arr.size else 0
        return _normalize(arr)


class SentenceTransformersEmbedder(Embedder):
    def __init__(self, settings: Settings):
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(settings.sentence_transformers_model)
        self.dim = self._model.get_sentence_embedding_dimension()

    def embed(self, texts: list[str]) -> np.ndarray:
        arr = self._model.encode(
            texts, convert_to_numpy=True, normalize_embeddings=False
        ).astype("float32")
        return _normalize(arr)


class OpenAIEmbedder(Embedder):
    """Any OpenAI-compatible /embeddings endpoint (OpenAI, Jina, Gemini, Voyage…)."""

    def __init__(self, settings: Settings):
        if not settings.embeddings_api_key:
            raise EmbeddingError(
                "EMBEDDINGS_API_KEY is not set but embeddings_provider=openai"
            )
        self._url = settings.embeddings_base_url.rstrip("/") + "/embeddings"
        self._key = settings.embeddings_api_key
        self._model = settings.embeddings_model
        self._timeout = settings.request_timeout
        self.dim = 0  # discovered on first call

    def embed(self, texts: list[str]) -> np.ndarray:
        headers = {"Authorization": f"Bearer {self._key}"}
        try:
            with httpx.Client(timeout=self._timeout) as client:
                resp = client.post(
                    self._url,
                    json={"model": self._model, "input": texts},
                    headers=headers,
                )
                resp.raise_for_status()
                data = resp.json()["data"]
        except httpx.HTTPError as exc:
            raise EmbeddingError(
                f"Embedding API request failed ({self._model}): {exc}"
            ) from exc
        # Preserve input order (OpenAI returns an "index" per item).
        data = sorted(data, key=lambda d: d.get("index", 0))
        arr = np.asarray([d["embedding"] for d in data], dtype="float32")
        self.dim = arr.shape[1] if arr.size else 0
        return _normalize(arr)


def _normalize(arr: np.ndarray) -> np.ndarray:
    """L2-normalize rows so an inner product equals cosine similarity."""
    if arr.size == 0:
        return arr
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


def build_embedder(settings: Settings) -> Embedder:
    provider = settings.embeddings_provider.lower()
    if provider == "ollama":
        return OllamaEmbedder(settings)
    if provider == "sentence_transformers":
        return SentenceTransformersEmbedder(settings)
    if provider == "openai":
        return OpenAIEmbedder(settings)
    raise EmbeddingError(f"Unknown embeddings_provider: {settings.embeddings_provider}")
