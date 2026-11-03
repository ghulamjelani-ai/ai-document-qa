"""Tests for the pure retrieval metrics."""
from __future__ import annotations

from app.metrics import (
    aggregate,
    hit_at_k,
    precision_at_k,
    reciprocal_rank,
    relevance_flags,
)


def test_relevance_flags_substring_case_insensitive():
    flags = relevance_flags(["Refund within 14 DAYS", "delivery fee"], ["14 days"])
    assert flags == [True, False]


def test_hit_and_precision_and_mrr():
    flags = [False, True, False, True]  # first relevant at rank 2
    assert hit_at_k(flags, 5) == 1.0
    assert hit_at_k(flags, 1) == 0.0
    assert precision_at_k(flags, 4) == 0.5
    assert reciprocal_rank(flags) == 0.5


def test_no_relevant_scores_zero():
    flags = [False, False, False]
    assert hit_at_k(flags, 3) == 0.0
    assert reciprocal_rank(flags) == 0.0


def test_aggregate_means():
    agg = aggregate([[True, False], [False, True]], k=2)
    assert agg["recall@k"] == 1.0           # both have a relevant chunk in top-2
    assert agg["mrr"] == (1.0 + 0.5) / 2     # ranks 1 and 2
    assert agg["n"] == 2
