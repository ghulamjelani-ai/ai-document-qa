"""Tests for the FAISS-backed vector store."""
from __future__ import annotations

import numpy as np

from app.core.vectorstore import VectorStore


def _vec(values):
    v = np.asarray([values], dtype="float32")
    return v / np.linalg.norm(v)


def _metas(n):
    return [{"chunk_index": i, "page": None, "text": f"chunk {i}"} for i in range(n)]


def test_add_search_delete(tmp_path):
    store = VectorStore(tmp_path)
    vecs = np.vstack([_vec([1, 0, 0]), _vec([0, 1, 0]), _vec([0, 0, 1])])
    doc_id = store.add("a.txt", vecs, _metas(3))

    assert store.stats() == {"documents": 1, "chunks": 3}

    hits = store.search(_vec([1, 0, 0])[0], top_k=1)
    assert len(hits) == 1
    assert hits[0].text == "chunk 0"
    assert hits[0].score > 0.99

    removed = store.delete(doc_id)
    assert removed == 3
    assert store.stats() == {"documents": 0, "chunks": 0}
    assert store.search(_vec([1, 0, 0])[0], top_k=3) == []


def test_persistence_across_instances(tmp_path):
    store = VectorStore(tmp_path)
    store.add("a.txt", np.vstack([_vec([1, 0, 0])]), _metas(1))

    reopened = VectorStore(tmp_path)
    assert reopened.stats() == {"documents": 1, "chunks": 1}
    hits = reopened.search(_vec([1, 0, 0])[0], top_k=1)
    assert len(hits) == 1


def test_delete_missing_returns_zero(tmp_path):
    store = VectorStore(tmp_path)
    store.add("a.txt", np.vstack([_vec([1, 0, 0])]), _metas(1))
    assert store.delete("does-not-exist") == 0


def test_documents_aggregation(tmp_path):
    store = VectorStore(tmp_path)
    store.add("a.txt", np.vstack([_vec([1, 0, 0]), _vec([0, 1, 0])]), _metas(2))
    docs = store.documents()
    assert len(docs) == 1
    assert docs[0]["chunks"] == 2
