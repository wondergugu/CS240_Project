"""Weighted coverage utility f(S) and marginal gain (Eq. 4 & 8 from paper)."""

from .types import ConceptUniverse, Document


class WeightedCoverageObjective:

    def __init__(self, universe: ConceptUniverse):
        self.universe = universe

    def evaluate(self, selected: list[Document]) -> float:
        covered = set()
        for doc in selected:
            concepts = self.universe.doc_concepts.get(doc.doc_id, set())
            covered |= concepts
        return sum(
            self.universe.concept_weights.get(u, 0.0) for u in covered
        )

    def marginal_gain(self, doc: Document, covered: set) -> float:
        doc_concepts = self.universe.doc_concepts.get(doc.doc_id, set())
        new_concepts = doc_concepts - covered
        return sum(
            self.universe.concept_weights.get(u, 0.0) for u in new_concepts
        )

    def covered_concepts(self, selected: list[Document]) -> set:
        covered = set()
        for doc in selected:
            covered |= self.universe.doc_concepts.get(doc.doc_id, set())
        return covered
