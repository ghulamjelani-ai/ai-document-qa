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


def _text_metas(texts):
    return [
        {"chunk_index": i, "page": None, "text": t} for i, t in enumerate(texts)
    ]


def test_hybrid_keyword_match(tmp_path):
    store = VectorStore(tmp_path)
    vecs = np.vstack([_vec([1, 0, 0]), _vec([0, 1, 0]), _vec([0, 0, 1])])
    store.add(
        "handbook.txt",
        vecs,
        _text_metas(
            [
                "Refunds are paid within fourteen days",
                "Delivery to Lahore takes two days",
                "We accept JazzCash and EasyPaisa payments",
            ]
        ),
    )
    # A keyword-heavy query (exact term) should surface the payments chunk and
    # flag it as a keyword match, even though its vector is not the closest.
    hits = store.hybrid_search("jazzcash", _vec([1, 0, 0])[0], top_k=3, mode="hybrid")
    assert any("JazzCash" in h.text and h.keyword_match for h in hits)


def test_hybrid_source_filter(tmp_path):
    store = VectorStore(tmp_path)
    store.add("a.txt", np.vstack([_vec([1, 0, 0])]), _text_metas(["refund policy"]))
    store.add("b.txt", np.vstack([_vec([0, 1, 0])]), _text_metas(["delivery policy"]))
    hits = store.hybrid_search(
        "policy", _vec([1, 1, 0])[0], top_k=5, mode="hybrid", source="b.txt"
    )
    assert hits and all(h.filename == "b.txt" for h in hits)


def test_keyword_only_mode(tmp_path):
    store = VectorStore(tmp_path)
    # Three chunks so BM25's IDF stays positive for a term in a single chunk.
    store.add(
        "a.txt",
        np.vstack([_vec([1, 0, 0]), _vec([0, 1, 0]), _vec([0, 0, 1])]),
        _text_metas(
            ["warranty repair claim", "loyalty points voucher", "delivery lahore fee"]
        ),
    )
    hits = store.hybrid_search("warranty", _vec([0, 1, 0])[0], top_k=3, mode="keyword")
    assert hits[0].text.startswith("warranty")
