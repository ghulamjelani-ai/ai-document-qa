"""A small persistent vector store with hybrid (keyword + vector) search.

Vectors (L2-normalized) are searched with inner product, which equals cosine
similarity. In parallel, a BM25 index searches the chunk text by keywords.
Hybrid search fuses both rankings with Reciprocal Rank Fusion (RRF), so a chunk
that ranks well in either list rises to the top — meaning + exact terms.

Vectors and metadata are persisted to disk (.npy + .json) so the index survives
restarts. BM25 is rebuilt from the stored text on load. Deletion filters and
rebuilds, which is simple and correct for portfolio-scale data.
"""
from __future__ import annotations

import json
import re
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


@dataclass
class SearchHit:
    score: float            # cosine similarity (for display and relevance)
    doc_id: str
    filename: str
    chunk_index: int
    page: int | None
    text: str
    keyword_match: bool = False  # True if this chunk also matched the BM25 query
    fused_score: float = 0.0     # RRF score when hybrid search was used


class VectorStore:
    def __init__(self, index_dir: Path):
        self._dir = Path(index_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._vectors_path = self._dir / "vectors.npy"
        self._meta_path = self._dir / "meta.json"
        self._lock = threading.Lock()
        self._vectors: np.ndarray = np.zeros((0, 0), dtype="float32")
        self._metas: list[dict] = []
        self._index = None   # lazy faiss index (pure vector search)
        self._bm25 = None    # lazy BM25 index (keyword search)
        self._load()

    # ---------- persistence ----------
    def _load(self) -> None:
        if self._vectors_path.exists() and self._meta_path.exists():
            self._vectors = np.load(self._vectors_path)
            self._metas = json.loads(self._meta_path.read_text(encoding="utf-8"))
            self._rebuild()

    def _save(self) -> None:
        np.save(self._vectors_path, self._vectors)
        self._meta_path.write_text(
            json.dumps(self._metas, ensure_ascii=False), encoding="utf-8"
        )

    def _rebuild(self) -> None:
        self._rebuild_index()
        self._rebuild_bm25()

    def _rebuild_index(self) -> None:
        if self._vectors.shape[0] == 0 or self._vectors.shape[1] == 0:
            self._index = None
            return
        import faiss

        index = faiss.IndexFlatIP(self._vectors.shape[1])
        index.add(self._vectors)
        self._index = index

    def _rebuild_bm25(self) -> None:
        if not self._metas:
            self._bm25 = None
            return
        from rank_bm25 import BM25Okapi

        corpus = [_tokenize(m["text"]) for m in self._metas]
        self._bm25 = BM25Okapi(corpus)

    # ---------- writes ----------
    def add(self, filename: str, vectors: np.ndarray, chunk_metas: list[dict]) -> str:
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
            self._rebuild()
            self._save()
        return doc_id

    def delete(self, doc_id: str) -> int:
        with self._lock:
            keep = [i for i, m in enumerate(self._metas) if m["doc_id"] != doc_id]
            removed = len(self._metas) - len(keep)
            if removed == 0:
                return 0
            self._vectors = (
                self._vectors[keep] if keep else np.zeros((0, 0), dtype="float32")
            )
            self._metas = [self._metas[i] for i in keep]
            self._rebuild()
            self._save()
        return removed

    def clear(self) -> None:
        with self._lock:
            self._vectors = np.zeros((0, 0), dtype="float32")
            self._metas = []
            self._index = None
            self._bm25 = None
            self._save()

    # ---------- reads ----------
    def search(self, query_vector: np.ndarray, top_k: int) -> list[SearchHit]:
        """Pure vector (cosine) search."""
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
            hits.append(self._hit(int(i), cosine=float(score)))
        return hits

    def hybrid_search(
        self,
        query_text: str,
        query_vector: np.ndarray,
        top_k: int,
        mode: str = "hybrid",
        rrf_k: int = 60,
        pool: int = 20,
        source: str | None = None,
    ) -> list[SearchHit]:
        """Fuse vector and BM25 rankings with Reciprocal Rank Fusion.

        mode: "hybrid" (both), "vector" (cosine only) or "keyword" (BM25 only).
        source: optional filename filter — only chunks from that document qualify.
        """
        with self._lock:
            n = len(self._metas)
            if n == 0:
                return []
            allowed = {
                i for i, m in enumerate(self._metas)
                if source is None or m["filename"] == source
            }
            if not allowed:
                return []

            qv = query_vector.astype("float32").reshape(-1)
            cosines = self._vectors @ qv  # exact cosine per chunk (vectors normalized)

            # Vector ranking
            vec_rank: dict[int, int] = {}
            if mode in ("hybrid", "vector"):
                order = [i for i in np.argsort(-cosines) if i in allowed][:pool]
                vec_rank = {int(i): r for r, i in enumerate(order)}

            # BM25 keyword ranking
            bm25_rank: dict[int, int] = {}
            if mode in ("hybrid", "keyword") and self._bm25 is not None:
                bm = self._bm25.get_scores(_tokenize(query_text))
                order = [
                    i for i in np.argsort(-bm)
                    if i in allowed and bm[i] > 0
                ][:pool]
                bm25_rank = {int(i): r for r, i in enumerate(order)}

            candidates = set(vec_rank) | set(bm25_rank)
            if not candidates:
                return []

            fused: dict[int, float] = {}
            for i in candidates:
                s = 0.0
                if i in vec_rank:
                    s += 1.0 / (rrf_k + vec_rank[i])
                if i in bm25_rank:
                    s += 1.0 / (rrf_k + bm25_rank[i])
                fused[i] = s

            ordered = sorted(candidates, key=lambda i: fused[i], reverse=True)[:top_k]
            return [
                self._hit(
                    i,
                    cosine=float(cosines[i]),
                    keyword_match=i in bm25_rank,
                    fused_score=fused[i],
                )
                for i in ordered
            ]

    def _hit(
        self, i: int, cosine: float, keyword_match: bool = False, fused_score: float = 0.0
    ) -> SearchHit:
        m = self._metas[i]
        return SearchHit(
            score=cosine,
            doc_id=m["doc_id"],
            filename=m["filename"],
            chunk_index=m["chunk_index"],
            page=m.get("page"),
            text=m["text"],
            keyword_match=keyword_match,
            fused_score=fused_score,
        )

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
