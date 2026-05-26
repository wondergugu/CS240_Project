"""Greedy (Rel/Cost) baseline: select by relevance per token."""

from ..types import Document


class GreedyRelCostSelector:

    def select(
        self, candidates: list[Document], budget: int
    ) -> list[Document]:
        remaining = list(candidates)
        selected = []
        used = 0
        while remaining:
            best = None
            best_density = -1.0
            for doc in remaining:
                if used + doc.token_count > budget:
                    continue
                density = doc.relevance_score / doc.token_count
                if density > best_density:
                    best_density = density
                    best = doc
            if best is None:
                break
            selected.append(best)
            used += best.token_count
            remaining.remove(best)
        return selected
