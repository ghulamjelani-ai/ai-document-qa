"""Streamlit UI for the AI Document Q&A backend.

Talks to the FastAPI backend over HTTP. Set API_BASE_URL to point at the backend
(defaults to http://localhost:8000).
"""
from __future__ import annotations

import os

import httpx
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")

st.set_page_config(page_title="AI Document Q&A", page_icon="📄", layout="wide")


def api(method: str, path: str, **kwargs):
    url = f"{API_BASE_URL}{path}"
    try:
        resp = httpx.request(method, url, timeout=180, **kwargs)
    except httpx.HTTPError as exc:
        st.error(f"Cannot reach the backend at {API_BASE_URL}. Is it running?\n\n{exc}")
        st.stop()
    if resp.status_code >= 400:
        try:
            msg = resp.json()["error"]["message"]
        except Exception:
            msg = resp.text
        st.error(f"Error {resp.status_code}: {msg}")
        return None
    return resp.json()


# ---------------- Sidebar: documents ----------------
with st.sidebar:
    st.title("📄 Documents")
    st.caption(f"Backend: {API_BASE_URL}")

    uploaded = st.file_uploader(
        "Upload PDF / TXT / MD",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
    )
    if uploaded and st.button("Ingest", type="primary", use_container_width=True):
        files = [("files", (f.name, f.getvalue())) for f in uploaded]
        with st.spinner("Reading, chunking and embedding…"):
            result = api("POST", "/documents/upload", files=files)
        if result:
            st.success(f"Added {result['total_chunks']} chunks.")

    st.divider()
    docs = api("GET", "/documents") or []
    if docs:
        st.subheader("Indexed")
        for d in docs:
            col1, col2 = st.columns([4, 1])
            col1.write(f"**{d['filename']}**  \n{d['chunks']} chunks")
            if col2.button("🗑", key=f"del-{d['doc_id']}", help="Delete"):
                api("DELETE", f"/documents/{d['doc_id']}")
                st.rerun()
    else:
        st.info("No documents yet. Upload one to start.")


# ---------------- Main: chat ----------------
st.title("Ask your documents")
st.caption("Answers are grounded only in your uploaded documents, with cited sources.")

if "history" not in st.session_state:
    st.session_state.history = []

for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])
        if turn.get("sources"):
            with st.expander(f"Sources ({len(turn['sources'])})"):
                for i, s in enumerate(turn["sources"], start=1):
                    page = f", page {s['page']}" if s.get("page") else ""
                    st.markdown(
                        f"**[{i}] {s['filename']}**{page} · score {s['score']}  \n"
                        f"> {s['snippet']}"
                    )

question = st.chat_input("Ask a question about your documents…")
if question:
    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Searching and answering…"):
            result = api("POST", "/chat/ask", json={"question": question})
        if result:
            st.markdown(result["answer"])
            badge = "grounded in context" if result["used_context"] else "no relevant context"
            st.caption(f"Model: {result['model']} · {badge}")
            if result["sources"]:
                with st.expander(f"Sources ({len(result['sources'])})"):
                    for i, s in enumerate(result["sources"], start=1):
                        page = f", page {s['page']}" if s.get("page") else ""
                        st.markdown(
                            f"**[{i}] {s['filename']}**{page} · score {s['score']}  \n"
                            f"> {s['snippet']}"
                        )
            st.session_state.history.append(
                {
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result["sources"],
                }
            )
