"""S-RAG selection algorithms for weighted coverage under a token budget.

`select` implements the fast density-greedy system variant described in
Section 4.1. `select_theoretical` implements Algorithm 1: best feasible
solution of size at most two plus every feasible size-three seed followed by
density-greedy completion.
"""

import heapq
from itertools import combinations

from ..objective import WeightedCoverageObjective
from ..types import ConceptUniverse, Document
from ..utils import total_cost, validate_budget, validate_candidates


class SRAGSelector:
    def __init__(self, universe: ConceptUniverse, use_lazy: bool = True):
        self.universe = universe
        self.objective = WeightedCoverageObjective(universe)
        self.use_lazy = use_lazy

    def select(
        self, candidates: list[Document], budget: int
    ) -> list[Document]:
        """Run the fast practical density-greedy variant."""
        documents = self._prepare(candidates, budget)
        return self._density_greedy(documents, budget)

    def select_theoretical(
        self, candidates: list[Document], budget: int
    ) -> list[Document]:
        """Run the partial-enumeration algorithm stated as Algorithm 1."""
        documents = self._prepare(candidates, budget)
        feasible = [doc for doc in documents if doc.token_count <= budget]
        best = self._best_small_solution(feasible, budget)

        for seed in combinations(feasible, 3):
            if total_cost(seed) > budget:
                continue
            completed = self._density_greedy(
                documents, budget, initial=list(seed)
            )
            best = self._better_solution(best, completed)
        return best

    def select_full(
        self, candidates: list[Document], budget: int
    ) -> list[Document]:
        """Backward-compatible name for the theoretical Algorithm 1 variant."""
        return self.select_theoretical(candidates, budget)

    def _prepare(
        self, candidates: list[Document], budget: int
    ) -> list[Document]:
        validate_budget(budget)
        return validate_candidates(candidates)

    def _best_small_solution(
        self, candidates: list[Document], budget: int
    ) -> list[Document]:
        best: list[Document] = []
        for size in (1, 2):
            for candidate in combinations(candidates, size):
                if total_cost(candidate) <= budget:
                    best = self._better_solution(best, list(candidate))
        return best

    def _better_solution(
        self, current: list[Document], candidate: list[Document]
    ) -> list[Document]:
        current_value = self.objective.evaluate(current)
        candidate_value = self.objective.evaluate(candidate)
        if candidate_value > current_value:
            return candidate
        if candidate_value < current_value:
            return current

        current_key = (
            total_cost(current),
            tuple(sorted(doc.doc_id for doc in current)),
        )
        candidate_key = (
            total_cost(candidate),
            tuple(sorted(doc.doc_id for doc in candidate)),
        )
        return candidate if candidate_key < current_key else current

    def _density_greedy(
        self,
        candidates: list[Document],
        budget: int,
        initial: list[Document] | None = None,
    ) -> list[Document]:
        if self.use_lazy:
            return self._density_greedy_lazy(candidates, budget, initial)
        return self._density_greedy_naive(candidates, budget, initial)

    def _density_greedy_naive(
        self,
        candidates: list[Document],
        budget: int,
        initial: list[Document] | None = None,
    ) -> list[Document]:
        selected = list(initial or [])
        used = total_cost(selected)
        covered = self.objective.covered_concepts(selected)
        selected_ids = {doc.doc_id for doc in selected}
        remaining = [
            doc for doc in candidates if doc.doc_id not in selected_ids
        ]

        while True:
            scored: list[tuple[float, str, Document]] = []
            for doc in remaining:
                if used + doc.token_count > budget:
                    continue
                gain = self.objective.marginal_gain(doc, covered)
                if gain > 0:
                    scored.append((-(gain / doc.token_count), doc.doc_id, doc))
            if not scored:
                return selected

            _, _, best = min(scored)
            selected.append(best)
            used += best.token_count
            covered |= self.universe.doc_concepts.get(
                best.doc_id, frozenset()
            )
            remaining.remove(best)

    def _density_greedy_lazy(
        self,
        candidates: list[Document],
        budget: int,
        initial: list[Document] | None = None,
    ) -> list[Document]:
        """Use diminishing-return upper bounds for lazy marginal evaluation."""
        selected = list(initial or [])
        used = total_cost(selected)
        covered = self.objective.covered_concepts(selected)
        selected_ids = {doc.doc_id for doc in selected}

        heap: list[tuple[float, str, Document]] = []
        for doc in candidates:
            if doc.doc_id in selected_ids or used + doc.token_count > budget:
                continue
            gain = self.objective.marginal_gain(doc, covered)
            if gain > 0:
                heapq.heappush(
                    heap, (-(gain / doc.token_count), doc.doc_id, doc)
                )

        while heap:
            _, _, doc = heapq.heappop(heap)
            if used + doc.token_count > budget:
                continue

            gain = self.objective.marginal_gain(doc, covered)
            if gain <= 0:
                continue
            updated = (-(gain / doc.token_count), doc.doc_id, doc)

            if heap and updated[:2] > heap[0][:2]:
                heapq.heappush(heap, updated)
                continue

            selected.append(doc)
            used += doc.token_count
            covered |= self.universe.doc_concepts.get(
                doc.doc_id, frozenset()
            )

        return selected
