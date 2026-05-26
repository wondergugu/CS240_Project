"""MMR (Maximal Marginal Relevance) baseline.

MMR score: lambda * rel(d, q) - (1 - lambda) * max_{d' in S} sim(d, d')
Uses Jaccard similarity over concept sets.
"""

from ..types import Document, ConceptUniverse


class MMRSelector:

    def __init__(self, universe: ConceptUniverse, lam: float = 0.6):
        self.universe = universe
        self.lam = lam

    def _jaccard(self, doc_a: Document, doc_b: Document) -> float:
        a = self.universe.doc_concepts.get(doc_a.doc_id, set())
        b = self.universe.doc_concepts.get(doc_b.doc_id, set())
        if not a and not b:
            return 0.0
        intersection = len(a & b)
        union = len(a | b)
        return intersection / union if union > 0 else 0.0

    def select(
        self, candidates: list[Document], budget: int
    ) -> list[Document]:
        remaining = list(candidates)
        selected: list[Document] = []
        used = 0

        while remaining:
            best = None
            best_score = float("-inf")
            for doc in remaining:
                if used + doc.token_count > budget:
                    continue
                relevance = doc.relevance_score
                if selected:
                    max_sim = max(self._jaccard(doc, s) for s in selected)
                else:
                    max_sim = 0.0
                score = self.lam * relevance - (1 - self.lam) * max_sim
                if score > best_score:
                    best_score = score
                    best = doc
            if best is None:
                break
            selected.append(best)
            used += best.token_count
            remaining.remove(best)
        return selected
