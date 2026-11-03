"""A small persistent vector store backed by FAISS.

Vectors (L2-normalized) are searched with inner product, which equals cosine
similarity. Vectors and their metadata are persisted to disk as .npy + .json so
the index survives restarts. Deletion is supported by filtering and rebuilding
the flat index, which is simple and correct for portfolio-scale data.
"""
from __future__ import annotations

import json
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class SearchHit:
    score: float
    doc_id: str
    filename: str
    chunk_index: int
    page: int | None
    text: str


class VectorStore:
    def __init__(self, index_dir: Path):
        self._dir = Path(index_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._vectors_path = self._dir / "vectors.npy"
        self._meta_path = self._dir / "meta.json"
        self._lock = threading.Lock()
        self._vectors: np.ndarray = np.zeros((0, 0), dtype="float32")
        self._metas: list[dict] = []
        self._index = None  # lazy faiss index
        self._load()

    # ---------- persistence ----------
    def _load(self) -> None:
        if self._vectors_path.exists() and self._meta_path.exists():
            self._vectors = np.load(self._vectors_path)
            self._metas = json.loads(self._meta_path.read_text(encoding="utf-8"))
            self._rebuild_index()

    def _save(self) -> None:
        np.save(self._vectors_path, self._vectors)
        self._meta_path.write_text(
            json.dumps(self._metas, ensure_ascii=False), encoding="utf-8"
        )

    def _rebuild_index(self) -> None:
        if self._vectors.shape[0] == 0 or self._vectors.shape[1] == 0:
            self._index = None
            return
        import faiss

        index = faiss.IndexFlatIP(self._vectors.shape[1])
        index.add(self._vectors)
        self._index = index

    # ---------- writes ----------
    def add(
        self, filename: str, vectors: np.ndarray, chunk_metas: list[dict]
    ) -> str:
        """Add one document's chunks. Returns a new doc_id."""
        if vectors.shape[0] != len(chunk_metas):
            raise ValueError("vectors and chunk_metas length mismatch")
        doc_id = uuid.uuid4().hex[:12]
        with self._lock:
            if self._vectors.shape[1] and vectors.shape[1] != self._vectors.shape[1]:
                raise ValueError(
                    "Embedding dimension changed; clear the index before switching "
                    "embedding models."
                )
            if self._vectors.shape[0] == 0:
                self._vectors = vectors.astype("float32")
            else:
                self._vectors = np.vstack([self._vectors, vectors.astype("float32")])
            for meta in chunk_metas:
                self._metas.append({**meta, "doc_id": doc_id, "filename": filename})
            self._rebuild_index()
            self._save()
        return doc_id

    def delete(self, doc_id: str) -> int:
        """Remove every chunk belonging to doc_id. Returns how many were removed."""
        with self._lock:
            keep = [i for i, m in enumerate(self._metas) if m["doc_id"] != doc_id]
            removed = len(self._metas) - len(keep)
            if removed == 0:
                return 0
            self._vectors = (
                self._vectors[keep] if keep else np.zeros((0, 0), dtype="float32")
            )
            self._metas = [self._metas[i] for i in keep]
            self._rebuild_index()
            self._save()
        return removed

    def clear(self) -> None:
        with self._lock:
            self._vectors = np.zeros((0, 0), dtype="float32")
            self._metas = []
            self._index = None
            self._save()

    # ---------- reads ----------
    def search(self, query_vector: np.ndarray, top_k: int) -> list[SearchHit]:
        with self._lock:
            if self._index is None or not self._metas:
                return []
            q = query_vector.astype("float32").reshape(1, -1)
            k = min(top_k, len(self._metas))
            scores, idxs = self._index.search(q, k)
        hits: list[SearchHit] = []
        for score, i in zip(scores[0], idxs[0]):
            if i < 0:
                continue
            m = self._metas[i]
            hits.append(
                SearchHit(
                    score=float(score),
                    doc_id=m["doc_id"],
                    filename=m["filename"],
                    chunk_index=m["chunk_index"],
                    page=m.get("page"),
                    text=m["text"],
                )
            )
        return hits

    def documents(self) -> list[dict]:
        with self._lock:
            agg: dict[str, dict] = {}
            for m in self._metas:
                d = agg.setdefault(
                    m["doc_id"],
                    {
                        "doc_id": m["doc_id"],
                        "filename": m["filename"],
                        "chunks": 0,
                        "characters": 0,
                    },
                )
                d["chunks"] += 1
                d["characters"] += len(m["text"])
            return list(agg.values())

    def stats(self) -> dict:
        with self._lock:
            return {
                "documents": len({m["doc_id"] for m in self._metas}),
                "chunks": len(self._metas),
            }
