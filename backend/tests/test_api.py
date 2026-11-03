"""End-to-end API tests using fake embedder/LLM (no Ollama needed)."""
from __future__ import annotations


def test_root_and_health(client):
    assert client.get("/").status_code == 200
    h = client.get("/health").json()
    assert h["status"] == "ok"
    assert h["documents"] == 0


def test_ask_without_documents_is_409(client):
    r = client.post("/chat/ask", json={"question": "hello?"})
    assert r.status_code == 409
    assert r.json()["error"]["code"] == "http_409"


def test_upload_then_ask_then_delete(client):
    content = b"The capital of France is Paris. The Eiffel Tower is in Paris."
    r = client.post(
        "/documents/upload",
        files={"files": ("geo.txt", content, "text/plain")},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["total_chunks"] >= 1
    doc_id = body["documents"][0]["doc_id"]

    # document is listed
    docs = client.get("/documents").json()
    assert any(d["doc_id"] == doc_id for d in docs)

    # ask -> grounded answer with sources
    r = client.post("/chat/ask", json={"question": "Where is the Eiffel Tower?"})
    assert r.status_code == 200, r.text
    ans = r.json()
    assert ans["used_context"] is True
    assert len(ans["sources"]) >= 1
    assert ans["model"] == "fake-model"

    # delete
    r = client.delete(f"/documents/{doc_id}")
    assert r.status_code == 200
    assert r.json()["chunks_removed"] >= 1
    assert client.get("/documents").json() == []


def test_unsupported_file_type_rejected(client):
    r = client.post(
        "/documents/upload",
        files={"files": ("pic.png", b"\x89PNG", "image/png")},
    )
    assert r.status_code == 415
