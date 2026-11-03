"""Evaluate retrieval quality on a labelled golden set.

Ingests the sample document, then runs each golden question through the
retriever under three modes (vector-only, keyword-only, hybrid) and reports
Recall@k, Precision@k and MRR for each. This is the table the README shows and
the clearest proof that the retrieval actually works — and that hybrid search
helps.

Run from the backend/ directory (needs a local Ollama for embeddings):

    uv run python eval/run_eval.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

# Make the `app` package importable when run as a script.
BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

from app.config import get_settings  # noqa: E402
from app.core.rag import RagEngine  # noqa: E402
from app.metrics import aggregate, relevance_flags  # noqa: E402

EVAL_DIR = Path(__file__).resolve().parent
K = 5
MODES = ["vector", "keyword", "hybrid"]


def main() -> int:
    gold = json.loads((EVAL_DIR / "golden_set.json").read_text(encoding="utf-8"))
    doc_path = EVAL_DIR / gold["document"]
    questions = gold["questions"]

    # Fresh, throwaway index so the real app data is untouched. Smaller chunks
    # (roughly one section each) make retrieval selective, so k=5 is meaningful
    # and the metrics can tell the modes apart.
    import os

    os.environ["DATA_DIR"] = tempfile.mkdtemp(prefix="aidocqa-eval-")
    os.environ["CHUNK_SIZE"] = "300"
    os.environ["CHUNK_OVERLAP"] = "50"
    get_settings.cache_clear()
    settings = get_settings()
    engine = RagEngine(settings)

    print(f"Ingesting {doc_path.name} …")
    info = engine.ingest_document(doc_path.name, doc_path.read_bytes())
    print(f"  {info['chunks']} chunks indexed\n")

    results: dict[str, dict] = {}
    for mode in MODES:
        per_question = []
        for item in questions:
            qvec = engine.embedder.embed([item["q"]])[0]
            hits = engine.store.hybrid_search(
                item["q"], qvec, top_k=K, mode=mode,
                rrf_k=settings.rrf_k, pool=settings.candidate_pool,
            )
            flags = relevance_flags([h.text for h in hits], item["expect"])
            per_question.append(flags)
        results[mode] = aggregate(per_question, K)

    # ---- table ----
    print(f"Retrieval evaluation · {len(questions)} questions · k={K}\n")
    header = f"{'mode':<10} {'Recall@'+str(K):>10} {'Precision@'+str(K):>12} {'MRR':>8}"
    print(header)
    print("-" * len(header))
    lines_md = [
        f"| Mode | Recall@{K} | Precision@{K} | MRR |",
        "| --- | ---: | ---: | ---: |",
    ]
    for mode in MODES:
        r = results[mode]
        print(f"{mode:<10} {r['recall@k']:>10.2f} {r['precision@k']:>12.2f} {r['mrr']:>8.2f}")
        lines_md.append(
            f"| {mode} | {r['recall@k']:.2f} | {r['precision@k']:.2f} | {r['mrr']:.2f} |"
        )

    (EVAL_DIR / "results.md").write_text(
        f"# Retrieval evaluation\n\n{len(questions)} questions · k={K} · "
        f"embeddings={settings.ollama_embedding_model}\n\n"
        + "\n".join(lines_md)
        + "\n",
        encoding="utf-8",
    )
    print(f"\nWrote {EVAL_DIR / 'results.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
