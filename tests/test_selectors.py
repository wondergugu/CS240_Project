import random
from itertools import combinations
from math import e

import pytest

from srag.objective import WeightedCoverageObjective
from srag.selectors import (
    GreedyRelCostSelector,
    MMRSelector,
    SRAGSelector,
    TopKSelector,
)
from srag.types import ConceptUniverse, Document
from srag.utils import total_cost


def _make_test_data(n_docs=20, n_concepts=30, budget=500, seed=42):
    rng = random.Random(seed)
    concepts = {f"c{i}" for i in range(n_concepts)}
    weights = {concept: rng.uniform(0.1, 1.0) for concept in concepts}
    docs = []
    doc_concepts = {}
    for i in range(n_docs):
        token_count = rng.randint(40, 150)
        doc = Document(
            doc_id=f"d{i}",
            text=f"doc {i}",
            relevance_score=1.0 / (1 + 0.1 * i) + rng.uniform(-0.05, 0.05),
            token_count=token_count,
        )
        docs.append(doc)
        n_covered = rng.randint(2, 8)
        doc_concepts[doc.doc_id] = set(
            rng.sample(sorted(concepts), n_covered)
        )
    universe = ConceptUniverse(
        concepts=concepts,
        concept_weights=weights,
        doc_concepts=doc_concepts,
    )
    return docs, universe, budget


@pytest.mark.parametrize(
    "make_selector",
    [
        lambda universe: TopKSelector(),
        lambda universe: MMRSelector(universe, lam=0.6),
        lambda universe: GreedyRelCostSelector(),
        lambda universe: SRAGSelector(universe),
    ],
)
def test_selectors_obey_budget(make_selector):
    docs, universe, budget = _make_test_data()
    selected = make_selector(universe).select(docs, budget)
    assert total_cost(selected) <= budget
    assert selected


def test_mmr_rejects_invalid_lambda():
    _, universe, _ = _make_test_data()
    with pytest.raises(ValueError, match="lam"):
        MMRSelector(universe, lam=1.1)


def test_selectors_reject_invalid_budget_and_duplicate_ids():
    docs, universe, _ = _make_test_data(n_docs=2)
    with pytest.raises(ValueError, match="budget"):
        SRAGSelector(universe).select(docs, -1)
    with pytest.raises(ValueError, match="duplicate document id"):
        TopKSelector().select([docs[0], docs[0]], 1000)


def test_srag_lazy_matches_naive_density_greedy():
    docs, universe, budget = _make_test_data(seed=19)
    lazy = SRAGSelector(universe, use_lazy=True).select(docs, budget)
    naive = SRAGSelector(universe, use_lazy=False).select(docs, budget)
    assert [doc.doc_id for doc in lazy] == [doc.doc_id for doc in naive]


def test_srag_theoretical_variant_reaches_observed_bound_on_small_instance():
    docs, universe, budget = _make_test_data(
        n_docs=8, n_concepts=15, budget=400, seed=99
    )
    objective = WeightedCoverageObjective(universe)

    optimum = 0.0
    for size in range(len(docs) + 1):
        for candidate in combinations(docs, size):
            if total_cost(candidate) <= budget:
                optimum = max(optimum, objective.evaluate(candidate))

    selected = SRAGSelector(universe, use_lazy=False).select_theoretical(
        docs, budget
    )
    ratio = objective.evaluate(selected) / optimum if optimum else 1.0
    assert ratio >= (1 - 1 / e) - 1e-12


def test_theoretical_variant_enumerates_size_three_seeds():
    concepts = {f"c{i}" for i in range(14)}
    weights = {concept: 1.0 for concept in concepts}
    docs = [
        Document("decoy", "decoy", 1.0, 4),
        Document("a", "a", 0.8, 3),
        Document("b", "b", 0.7, 3),
        Document("c", "c", 0.6, 3),
    ]
    universe = ConceptUniverse(
        concepts=concepts,
        concept_weights=weights,
        doc_concepts={
            "decoy": {"c0", "c1", "c2", "c3", "c4"},
            "a": {"c5", "c6", "c7"},
            "b": {"c8", "c9", "c10"},
            "c": {"c11", "c12", "c13"},
        },
    )
    objective = WeightedCoverageObjective(universe)
    selector = SRAGSelector(universe, use_lazy=False)

    fast = selector.select(docs, budget=9)
    theoretical = selector.select_theoretical(docs, budget=9)

    assert objective.evaluate(fast) == 8.0
    assert objective.evaluate(theoretical) == 9.0
    assert {doc.doc_id for doc in theoretical} == {"a", "b", "c"}


def test_theoretical_variant_considers_best_size_two_solution():
    docs = [
        Document("a", "a", 0.9, 5),
        Document("b", "b", 0.8, 5),
    ]
    universe = ConceptUniverse(
        concepts={"a", "b"},
        concept_weights={"a": 1.0, "b": 2.0},
        doc_concepts={"a": {"a"}, "b": {"b"}},
    )
    selected = SRAGSelector(universe).select_theoretical(docs, budget=10)
    assert {doc.doc_id for doc in selected} == {"a", "b"}


def test_srag_deterministic():
    docs, universe, budget = _make_test_data()
    selector = SRAGSelector(universe)
    result1 = selector.select(docs, budget)
    result2 = selector.select(docs, budget)
    assert [doc.doc_id for doc in result1] == [doc.doc_id for doc in result2]


def test_srag_outperforms_topk_on_redundant_data():
    concepts = {f"c{i}" for i in range(20)}
    weights = {concept: 1.0 for concept in concepts}
    docs = []
    doc_concepts = {}

    shared = {"c0", "c1", "c2", "c3", "c4"}
    for i in range(5):
        doc = Document(f"d{i}", f"doc {i}", 0.95 - i * 0.01, 100)
        docs.append(doc)
        doc_concepts[doc.doc_id] = shared.copy()

    for i in range(5, 15):
        doc = Document(f"d{i}", f"doc {i}", 0.5 - i * 0.01, 80)
        docs.append(doc)
        doc_concepts[doc.doc_id] = {
            f"c{i}",
            f"c{i + 1}" if i + 1 < 20 else "c0",
        }

    universe = ConceptUniverse(
        concepts=concepts,
        concept_weights=weights,
        doc_concepts=doc_concepts,
    )
    objective = WeightedCoverageObjective(universe)

    topk = TopKSelector().select(docs, budget=400)
    srag = SRAGSelector(universe).select(docs, budget=400)
    assert objective.evaluate(srag) >= objective.evaluate(topk)
