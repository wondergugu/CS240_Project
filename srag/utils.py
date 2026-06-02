"""Shared helpers for document costs and input validation."""

from collections.abc import Iterable

from .types import Document


def count_tokens(text: str) -> int:
    """Return the demo's whitespace-token approximation of passage cost."""
    return len(text.split())


def total_cost(documents: Iterable[Document]) -> int:
    """Return c(S), the sum of document token costs."""
    return sum(doc.token_count for doc in documents)


def validate_budget(budget: int) -> None:
    if budget < 0:
        raise ValueError("budget must be nonnegative")


def validate_candidates(candidates: Iterable[Document]) -> list[Document]:
    """Materialize candidates and enforce the document-ID mapping contract."""
    documents = list(candidates)
    seen: set[str] = set()
    for doc in documents:
        if doc.doc_id in seen:
            raise ValueError(f"duplicate document id: {doc.doc_id!r}")
        seen.add(doc.doc_id)
    return documents
