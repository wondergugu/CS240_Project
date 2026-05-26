import random
from itertools import combinations

from srag.types import Document, ConceptUniverse
from srag.objective import WeightedCoverageObjective
from srag.selectors import TopKSelector, MMRSelector, GreedyRelCostSelector, SRAGSelector


def _make_test_data(n_docs=20, n_concepts=30, budget=500, seed=42):
    rng = random.Random(seed)
    concepts = {f"c{i}" for i in range(n_concepts)}
    weights = {c: rng.uniform(0.1, 1.0) for c in concepts}
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
        doc_concepts[doc.doc_id] = set(rng.sample(sorted(concepts), n_covered))
    universe = ConceptUniverse(
        concepts=concepts, concept_weights=weights, doc_concepts=doc_concepts
    )
    return docs, universe, budget


def test_topk_budget_constraint():
    docs, universe, budget = _make_test_data()
    selector = TopKSelector()
    selected = selector.select(docs, budget)
    total_cost = sum(d.token_count for d in selected)
    assert total_cost <= budget
    assert len(selected) > 0


def test_mmr_budget_constraint():
    docs, universe, budget = _make_test_data()
    selector = MMRSelector(universe, lam=0.6)
    selected = selector.select(docs, budget)
    total_cost = sum(d.token_count for d in selected)
    assert total_cost <= budget
    assert len(selected) > 0


def test_greedy_relcost_budget_constraint():
    docs, universe, budget = _make_test_data()
    selector = GreedyRelCostSelector()
    selected = selector.select(docs, budget)
    total_cost = sum(d.token_count for d in selected)
    assert total_cost <= budget
    assert len(selected) > 0


def test_srag_budget_constraint():
    docs, universe, budget = _make_test_data()
    selector = SRAGSelector(universe)
    selected = selector.select(docs, budget)
    total_cost = sum(d.token_count for d in selected)
    assert total_cost <= budget
    assert len(selected) > 0


def test_srag_approximation_guarantee():
    """On small instances, verify S-RAG achieves >= (1-1/e) * OPT."""
    docs, universe, budget = _make_test_data(n_docs=8, n_concepts=15, budget=400, seed=99)
    obj = WeightedCoverageObjective(universe)

    # Brute-force optimal
    best_val = 0.0
    for size in range(1, len(docs) + 1):
        for combo in combinations(docs, size):
            cost = sum(d.token_count for d in combo)
            if cost > budget:
                continue
            val = obj.evaluate(list(combo))
            if val > best_val:
                best_val = val

    selector = SRAGSelector(universe, use_lazy=False)
    selected = selector.select(docs, budget)
    srag_val = obj.evaluate(selected)

    ratio = srag_val / best_val if best_val > 0 else 1.0
    assert ratio >= (1 - 1 / 2.718) - 0.01, (
        f"Approximation ratio {ratio:.3f} below (1-1/e)={1-1/2.718:.3f}"
    )


def test_srag_deterministic():
    docs, universe, budget = _make_test_data()
    selector = SRAGSelector(universe)
    result1 = selector.select(docs, budget)
    result2 = selector.select(docs, budget)
    assert [d.doc_id for d in result1] == [d.doc_id for d in result2]


def test_srag_outperforms_topk_on_redundant_data():
    """When top docs are redundant, S-RAG should achieve higher coverage."""
    rng = random.Random(777)
    concepts = {f"c{i}" for i in range(20)}
    weights = {c: 1.0 for c in concepts}
    docs = []
    doc_concepts = {}

    # Top 5 docs all cover the same concepts (redundant)
    shared = {"c0", "c1", "c2", "c3", "c4"}
    for i in range(5):
        doc = Document(f"d{i}", f"doc {i}", 0.95 - i * 0.01, 100)
        docs.append(doc)
        doc_concepts[doc.doc_id] = shared.copy()

    # Lower-ranked docs cover unique concepts
    for i in range(5, 15):
        doc = Document(f"d{i}", f"doc {i}", 0.5 - i * 0.01, 80)
        docs.append(doc)
        doc_concepts[doc.doc_id] = {f"c{i}", f"c{i+1}" if i + 1 < 20 else "c0"}

    universe = ConceptUniverse(
        concepts=concepts, concept_weights=weights, doc_concepts=doc_concepts
    )
    budget = 400
    obj = WeightedCoverageObjective(universe)

    topk_selected = TopKSelector().select(docs, budget)
    srag_selected = SRAGSelector(universe).select(docs, budget)

    topk_val = obj.evaluate(topk_selected)
    srag_val = obj.evaluate(srag_selected)
    assert srag_val >= topk_val
