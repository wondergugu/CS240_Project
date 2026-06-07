"""Tests for the retrieval->selection adapter (no model weights required)."""

import pytest

from rag.selection import Candidate, build_documents, select
from srag.utils import total_cost

_TEXTS = [
    ("d0", "climate change raises global average surface temperature", 0.9),
    ("d1", "carbon dioxide emissions trap heat in the atmosphere", 0.8),
    ("d2", "renewable solar and wind energy reduce emissions", 0.7),
    ("d3", "global temperature records broke again last summer", 0.6),
    ("d4", "ocean acidification harms coral reef ecosystems", 0.5),
]


def _candidates():
    return [Candidate(doc_id=i, text=t, relevance=r) for i, t, r in _TEXTS]


def _word_count(text: str) -> int:
    return len(text.split())


def test_build_documents_clamps_relevance_and_positive_tokens():
    docs = build_documents(
        [Candidate("x", "two words", -0.3)], _word_count
    )
    assert docs[0].relevance_score == 0.0
    assert docs[0].token_count == 2


@pytest.mark.parametrize("method", ["topk", "greedy", "mmr", "srag"])
def test_select_respects_budget(method):
    documents = build_documents(_candidates(), _word_count)
    budget = 12
    selected = select(method, documents, budget, top_l=5)
    assert total_cost(selected) <= budget
    assert len({d.doc_id for d in selected}) == len(selected)


def test_select_empty_and_unknown():
    assert select("srag", [], 100) == []
    docs = build_documents(_candidates(), _word_count)
    with pytest.raises(ValueError):
        select("nope", docs, 100)
