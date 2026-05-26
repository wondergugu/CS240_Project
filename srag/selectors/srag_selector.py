"""S-RAG selector: density-greedy with lazy evaluation (Algorithm 1).

Two variants:
- Fast: pure density-greedy (no partial enumeration), used in practice
- Full: partial enumeration over size-3 seeds + density-greedy completion
"""

import heapq
from itertools import combinations

from ..types import Document, ConceptUniverse
from ..objective import WeightedCoverageObjective


class SRAGSelector:

    def __init__(self, universe: ConceptUniverse, use_lazy: bool = True):
        self.universe = universe
        self.objective = WeightedCoverageObjective(universe)
        self.use_lazy = use_lazy

    def select(
        self, candidates: list[Document], budget: int
    ) -> list[Document]:
        return self._density_greedy(candidates, budget)

    def select_full(
        self, candidates: list[Document], budget: int, max_seed_size: int = 3
    ) -> list[Document]:
        """Theoretical variant with partial enumeration (Algorithm 1)."""
        best_solution = []
        best_value = 0.0

        feasible = [d for d in candidates if d.token_count <= budget]

        for size in range(1, min(max_seed_size + 1, 3)):
            for seed in combinations(feasible, size):
                seed_cost = sum(d.token_count for d in seed)
                if seed_cost > budget:
                    continue
                seed_list = list(seed)
                remaining = [d for d in candidates if d not in seed]
                completed = self._density_greedy(
                    remaining, budget, initial=seed_list
                )
                val = self.objective.evaluate(completed)
                if val > best_value:
                    best_value = val
                    best_solution = completed

        greedy_only = self._density_greedy(candidates, budget)
        greedy_val = self.objective.evaluate(greedy_only)
        if greedy_val > best_value:
            best_solution = greedy_only

        return best_solution

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
        selected = list(initial) if initial else []
        used = sum(d.token_count for d in selected)
        covered = self.objective.covered_concepts(selected)
        remaining = [d for d in candidates if d not in selected]

        while remaining:
            best = None
            best_density = -1.0
            for doc in remaining:
                if used + doc.token_count > budget:
                    continue
                gain = self.objective.marginal_gain(doc, covered)
                if gain <= 0:
                    continue
                density = gain / doc.token_count
                if density > best_density:
                    best_density = density
                    best = doc
            if best is None:
                break
            selected.append(best)
            used += best.token_count
            covered |= self.universe.doc_concepts.get(best.doc_id, set())
            remaining.remove(best)
        return selected

    def _density_greedy_lazy(
        self,
        candidates: list[Document],
        budget: int,
        initial: list[Document] | None = None,
    ) -> list[Document]:
        """Density-greedy with lazy evaluation (Minoux 1978)."""
        selected = list(initial) if initial else []
        used = sum(d.token_count for d in selected)
        covered = self.objective.covered_concepts(selected)
        selected_ids = {d.doc_id for d in selected}

        heap = []
        for doc in candidates:
            if doc.doc_id in selected_ids:
                continue
            if doc.token_count > budget:
                continue
            gain = self.objective.marginal_gain(doc, covered)
            if gain > 0:
                density = gain / doc.token_count
                heapq.heappush(heap, (-density, doc.doc_id, doc, True))

        while heap:
            neg_density, _, doc, is_current = heapq.heappop(heap)

            if used + doc.token_count > budget:
                continue

            if is_current:
                selected.append(doc)
                used += doc.token_count
                covered |= self.universe.doc_concepts.get(doc.doc_id, set())
                # Invalidate all remaining entries
                new_heap = []
                for item in heap:
                    _, _, d, _ = item
                    if used + d.token_count <= budget:
                        gain = self.objective.marginal_gain(d, covered)
                        if gain > 0:
                            density = gain / d.token_count
                            new_heap.append((-density, d.doc_id, d, True))
                heap = new_heap
                heapq.heapify(heap)
            else:
                gain = self.objective.marginal_gain(doc, covered)
                if gain > 0 and used + doc.token_count <= budget:
                    density = gain / doc.token_count
                    heapq.heappush(heap, (-density, doc.doc_id, doc, True))

        return selected
