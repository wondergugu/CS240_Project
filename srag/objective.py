"""Weighted coverage utility f(S) and marginal gain from Eq. (4)."""

from collections.abc import Iterable, Set
from math import fsum

from .types import ConceptUniverse, Document


class WeightedCoverageObjective:
    def __init__(self, universe: ConceptUniverse):
        self.universe = universe

    def evaluate(self, selected: Iterable[Document]) -> float:
        return fsum(
            self.universe.concept_weights[concept]
            for concept in self.covered_concepts(selected)
        )

    def marginal_gain(
        self, doc: Document, covered: Set[str]
    ) -> float:
        new_concepts = (
            self.universe.doc_concepts.get(doc.doc_id, frozenset()) - covered
        )
        return fsum(
            self.universe.concept_weights[concept]
            for concept in new_concepts
        )

    def covered_concepts(
        self, selected: Iterable[Document]
    ) -> set[str]:
        covered: set[str] = set()
        for doc in selected:
            covered.update(
                self.universe.doc_concepts.get(doc.doc_id, frozenset())
            )
        return covered
