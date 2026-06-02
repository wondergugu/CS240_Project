"""MMR baseline with a configurable inter-document similarity function.

The synthetic demo has no embedding model and therefore uses Jaccard
similarity over concept sets as a proxy. A real evaluation can inject its
embedding-based document similarity while retaining the MMR selection rule.
"""

from collections.abc import Callable

from ..types import ConceptUniverse, Document
from ..utils import validate_budget, validate_candidates


class MMRSelector:
    def __init__(
        self,
        universe: ConceptUniverse,
        lam: float = 0.6,
        similarity: Callable[[Document, Document], float] | None = None,
    ):
        if not 0.0 <= lam <= 1.0:
            raise ValueError("lam must be between 0 and 1")
        self.universe = universe
        self.lam = lam
        self.similarity = similarity or self._concept_jaccard

    def _concept_jaccard(self, doc_a: Document, doc_b: Document) -> float:
        a = self.universe.doc_concepts.get(doc_a.doc_id, frozenset())
        b = self.universe.doc_concepts.get(doc_b.doc_id, frozenset())
        if not a and not b:
            return 0.0
        return len(a & b) / len(a | b)

    def select(
        self, candidates: list[Document], budget: int
    ) -> list[Document]:
        validate_budget(budget)
        remaining = validate_candidates(candidates)
        selected: list[Document] = []
        used = 0

        while True:
            feasible = [
                doc
                for doc in remaining
                if used + doc.token_count <= budget
            ]
            if not feasible:
                return selected

            def ranking_key(doc: Document) -> tuple[float, float, str]:
                redundancy = (
                    max(self.similarity(doc, chosen) for chosen in selected)
                    if selected
                    else 0.0
                )
                score = (
                    self.lam * doc.relevance_score
                    - (1 - self.lam) * redundancy
                )
                return (-score, -doc.relevance_score, doc.doc_id)

            best = min(feasible, key=ranking_key)
            selected.append(best)
            used += best.token_count
            remaining.remove(best)
