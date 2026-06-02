import pytest

from srag.types import ConceptUniverse, Document


@pytest.mark.parametrize(
    ("relevance", "token_count", "message"),
    [
        (-0.1, 10, "relevance_score"),
        (0.5, 0, "token_count"),
    ],
)
def test_document_rejects_values_outside_paper_contract(
    relevance, token_count, message
):
    with pytest.raises(ValueError, match=message):
        Document("d0", "content", relevance, token_count)


def test_concept_universe_requires_one_weight_per_concept():
    with pytest.raises(ValueError, match="concept_weights"):
        ConceptUniverse(
            concepts={"known", "missing"},
            concept_weights={"known": 1.0},
            doc_concepts={"d0": {"known"}},
        )


def test_concept_universe_rejects_out_of_universe_coverage():
    with pytest.raises(ValueError, match="outside universe"):
        ConceptUniverse(
            concepts={"known"},
            concept_weights={"known": 1.0},
            doc_concepts={"d0": {"unknown"}},
        )
