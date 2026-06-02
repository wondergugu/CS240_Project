"""Greedy relevance-per-token baseline from the paper's comparisons."""

from ..types import Document
from ..utils import validate_budget, validate_candidates


class GreedyRelCostSelector:
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

            best = min(
                feasible,
                key=lambda doc: (
                    -(doc.relevance_score / doc.token_count),
                    -doc.relevance_score,
                    doc.doc_id,
                ),
            )
            selected.append(best)
            used += best.token_count
            remaining.remove(best)
