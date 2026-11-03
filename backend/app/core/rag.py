"""The RAG engine: ingestion and grounded question answering.

Ties together the embedder, the FAISS vector store and the LLM. Answers are
grounded strictly in retrieved chunks, every answer carries its sources, and
when nothing relevant is found the engine says so instead of guessing.
"""
from __future__ import annotations

from dataclasses import dataclass

from app.config import Settings
from app.core.embeddings import Embedder, build_embedder
from app.core.ingest import ingest
from app.core.llm import LLM, build_llm
from app.core.vectorstore import SearchHit, VectorStore

SYSTEM_PROMPT = (
    "You are a precise assistant that answers questions using ONLY the provided "
    "context from the user's documents. Follow these rules:\n"
    "- Use only facts found in the context. Do not use outside knowledge.\n"
    "- If the context does not contain the answer, reply exactly: "
    '"I could not find the answer in the provided documents."\n'
    "- Be concise and factual. When helpful, cite sources inline like [1], [2] "
    "matching the numbered context blocks.\n"
)

NO_ANSWER = "I could not find the answer in the provided documents."


@dataclass
class Answer:
    answer: str
    hits: list[SearchHit]
    used_context: bool
    model: str
    retrieval_mode: str


class RagEngine:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.store = VectorStore(settings.index_dir)
        self._embedder: Embedder | None = None
        self._llm: LLM | None = None

    # Providers are built lazily so the API can start even if Ollama is briefly
    # unreachable; the error then surfaces on the first real request.
    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = build_embedder(self.settings)
        return self._embedder

    @property
    def llm(self) -> LLM:
        if self._llm is None:
            self._llm = build_llm(self.settings)
        return self._llm

    # ---------- ingestion ----------
    def ingest_document(self, filename: str, data: bytes) -> dict:
        chunks = ingest(
            filename, data, self.settings.chunk_size, self.settings.chunk_overlap
        )
        if not chunks:
            raise ValueError("No extractable text found in the document.")
        vectors = self.embedder.embed([c.text for c in chunks])
        metas = [
            {"chunk_index": c.chunk_index, "page": c.page, "text": c.text}
            for c in chunks
        ]
        doc_id = self.store.add(filename, vectors, metas)
        return {
            "doc_id": doc_id,
            "filename": filename,
            "chunks": len(chunks),
            "characters": sum(len(c.text) for c in chunks),
        }

    # ---------- question answering ----------
    def answer(
        self, question: str, top_k: int | None = None, source: str | None = None
    ) -> Answer:
        k = top_k or self.settings.top_k
        query_vec = self.embedder.embed([question])[0]
        hits = self.store.hybrid_search(
            question,
            query_vec,
            top_k=k,
            mode=self.settings.retrieval_mode,
            rrf_k=self.settings.rrf_k,
            pool=self.settings.candidate_pool,
            source=source,
        )
        # A chunk counts as relevant if it is semantically close (cosine) OR it
        # matched the keyword search — this is the whole point of hybrid search,
        # where exact terms (codes, names) score low on cosine but high on BM25.
        relevant = [
            h for h in hits
            if h.score >= self.settings.score_threshold or h.keyword_match
        ]

        if not relevant:
            return Answer(
                answer=NO_ANSWER,
                hits=[],
                used_context=False,
                model=self.llm.name,
                retrieval_mode=self.settings.retrieval_mode,
            )

        context = "\n\n".join(
            f"[{i}] (from {h.filename}"
            + (f", page {h.page}" if h.page else "")
            + f")\n{h.text}"
            for i, h in enumerate(relevant, start=1)
        )
        user_prompt = (
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Answer using only the context above."
        )
        text = self.llm.complete(SYSTEM_PROMPT, user_prompt)
        return Answer(
            answer=text,
            hits=relevant,
            used_context=True,
            model=self.llm.name,
            retrieval_mode=self.settings.retrieval_mode,
        )
