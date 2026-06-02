"""Top-k baseline: pack candidates in descending retrieval relevance order."""

from ..types import Document
from ..utils import validate_budget, validate_candidates


class TopKSelector:
    def select(
        self, candidates: list[Document], budget: int
    ) -> list[Document]:
        validate_budget(budget)
        documents = validate_candidates(candidates)
        ranked = sorted(
            documents, key=lambda doc: (-doc.relevance_score, doc.doc_id)
        )

        selected: list[Document] = []
        used = 0
        for doc in ranked:
            if used + doc.token_count <= budget:
                selected.append(doc)
                used += doc.token_count
        return selected
