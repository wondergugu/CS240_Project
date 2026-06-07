"""Adapt retrieved candidates to the `srag` selectors under a token budget.

Each retrieved passage becomes a `srag.Document` carrying its retriever
relevance `rel(d, q)` (mapped to a nonnegative score) and a real generator
token cost. The concept universe and weighted-coverage objective are reused
unchanged from the selection core; only the inputs are now real.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from srag.concepts import build_concept_universe
from srag.selectors import (
    GreedyRelCostSelector,
    MMRSelector,
    SRAGSelector,
    TopKSelector,
)
from srag.types import Document

SimilarityFn = Callable[[Document, Document], float]
TokenCounter = Callable[[str], int]


@dataclass(frozen=True, slots=True)
class Candidate:
    """A retrieved passage with its retriever relevance score."""

    doc_id: str
    text: str
    relevance: float


def build_documents(
    candidates: Sequence[Candidate], token_counter: TokenCounter
) -> list[Document]:
    """Convert candidates to budget-aware `Document`s (relevance clamped >= 0)."""
    documents: list[Document] = []
    for cand in candidates:
        token_count = max(1, int(token_counter(cand.text)))
        documents.append(
            Document(
                doc_id=cand.doc_id,
                text=cand.text,
                relevance_score=max(0.0, float(cand.relevance)),
                token_count=token_count,
            )
        )
    return documents


def select(
    method: str,
    documents: list[Document],
    budget: int,
    *,
    top_l: int = 20,
    similarity: SimilarityFn | None = None,
    mmr_lambda: float = 0.5,
) -> list[Document]:
    """Run one selection strategy on real candidates under a token budget."""
    if not documents:
        return []

    if method == "topk":
        return TopKSelector().select(documents, budget)
    if method == "greedy":
        return GreedyRelCostSelector().select(documents, budget)

    universe = build_concept_universe(documents, top_l=min(top_l, len(documents)))
    if method == "mmr":
        return MMRSelector(
            universe, lam=mmr_lambda, similarity=similarity
        ).select(documents, budget)
    if method == "srag":
        return SRAGSelector(universe, use_lazy=True).select(documents, budget)

    raise ValueError(f"unknown selection method: {method!r}")
