import random
from itertools import combinations

from srag.types import Document, ConceptUniverse
from srag.objective import WeightedCoverageObjective


def _random_universe(n_docs=10, n_concepts=20, seed=42):
    rng = random.Random(seed)
    concepts = {f"c{i}" for i in range(n_concepts)}
    weights = {c: rng.uniform(0.1, 1.0) for c in concepts}
    docs = []
    doc_concepts = {}
    for i in range(n_docs):
        doc = Document(
            doc_id=f"d{i}",
            text=f"doc {i}",
            relevance_score=rng.uniform(0.1, 1.0),
            token_count=rng.randint(50, 200),
        )
        docs.append(doc)
        n_covered = rng.randint(1, min(8, n_concepts))
        doc_concepts[doc.doc_id] = set(rng.sample(sorted(concepts), n_covered))
    universe = ConceptUniverse(
        concepts=concepts,
        concept_weights=weights,
        doc_concepts=doc_concepts,
    )
    return docs, universe


def test_monotonicity():
    docs, universe = _random_universe(n_docs=8, seed=123)
    obj = WeightedCoverageObjective(universe)
    for size_s in range(len(docs)):
        for s_combo in combinations(docs, size_s):
            s_list = list(s_combo)
            f_s = obj.evaluate(s_list)
            for extra in docs:
                if extra not in s_combo:
                    f_t = obj.evaluate(s_list + [extra])
                    assert f_t >= f_s, (
                        f"Monotonicity violated: f(S)={f_s}, f(S+d)={f_t}"
                    )


def test_submodularity():
    docs, universe = _random_universe(n_docs=6, seed=456)
    obj = WeightedCoverageObjective(universe)
    for s_size in range(len(docs) - 1):
        for s_combo in combinations(range(len(docs)), s_size):
            s_set = set(s_combo)
            for t_size in range(s_size, len(docs)):
                for t_combo in combinations(range(len(docs)), t_size):
                    t_set = set(t_combo)
                    if not s_set.issubset(t_set):
                        continue
                    s_list = [docs[i] for i in s_combo]
                    t_list = [docs[i] for i in t_combo]
                    covered_s = obj.covered_concepts(s_list)
                    covered_t = obj.covered_concepts(t_list)
                    for d_idx in range(len(docs)):
                        if d_idx in t_set:
                            continue
                        d = docs[d_idx]
                        gain_s = obj.marginal_gain(d, covered_s)
                        gain_t = obj.marginal_gain(d, covered_t)
                        assert gain_s >= gain_t - 1e-10, (
                            f"Submodularity violated: "
                            f"gain(d|S)={gain_s}, gain(d|T)={gain_t}"
                        )


def test_non_negativity():
    docs, universe = _random_universe(seed=789)
    obj = WeightedCoverageObjective(universe)
    assert obj.evaluate([]) == 0.0
    for doc in docs:
        assert obj.evaluate([doc]) >= 0.0
    assert obj.evaluate(docs) >= 0.0


def test_marginal_gain_consistency():
    docs, universe = _random_universe(seed=101)
    obj = WeightedCoverageObjective(universe)
    selected = docs[:3]
    covered = obj.covered_concepts(selected)
    for doc in docs[3:]:
        gain = obj.marginal_gain(doc, covered)
        f_before = obj.evaluate(selected)
        f_after = obj.evaluate(selected + [doc])
        assert abs(gain - (f_after - f_before)) < 1e-10
