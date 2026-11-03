"""Pure retrieval-quality metrics. No LLM, no network — just ranking math.

A retrieved chunk is "relevant" to a question if its text contains any of the
question's expected substrings (case-insensitive). Given the relevance flags of
the top-k retrieved chunks (in rank order), these functions score the ranking.
"""
from __future__ import annotations


def relevance_flags(chunk_texts: list[str], expects: list[str]) -> list[bool]:
    needles = [e.lower() for e in expects]
    return [any(n in t.lower() for n in needles) for t in chunk_texts]


def hit_at_k(flags: list[bool], k: int) -> float:
    """1.0 if any of the top-k chunks is relevant, else 0.0 (a.k.a. recall@k
    when there is one relevant section per question)."""
    return 1.0 if any(flags[:k]) else 0.0


def precision_at_k(flags: list[bool], k: int) -> float:
    if k <= 0:
        return 0.0
    top = flags[:k]
    return sum(top) / k


def reciprocal_rank(flags: list[bool]) -> float:
    """1 / (rank of first relevant chunk); 0 if none are relevant."""
    for rank, is_rel in enumerate(flags, start=1):
        if is_rel:
            return 1.0 / rank
    return 0.0


def aggregate(per_question: list[list[bool]], k: int) -> dict[str, float]:
    """Mean metrics over a set of questions, each given as its ranked flags."""
    if not per_question:
        return {"recall@k": 0.0, "precision@k": 0.0, "mrr": 0.0, "k": k, "n": 0}
    n = len(per_question)
    return {
        "recall@k": sum(hit_at_k(f, k) for f in per_question) / n,
        "precision@k": sum(precision_at_k(f, k) for f in per_question) / n,
        "mrr": sum(reciprocal_rank(f) for f in per_question) / n,
        "k": k,
        "n": n,
    }
