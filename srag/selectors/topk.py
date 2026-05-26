"""Top-k baseline: select documents by relevance score until budget is filled."""

from ..types import Document


class TopKSelector:

    def select(
        self, candidates: list[Document], budget: int
    ) -> list[Document]:
        sorted_docs = sorted(
            candidates, key=lambda d: d.relevance_score, reverse=True
        )
        selected = []
        used = 0
        for doc in sorted_docs:
            if used + doc.token_count <= budget:
                selected.append(doc)
                used += doc.token_count
        return selected
