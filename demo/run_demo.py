"""End-to-end demo comparing all document selection methods."""

import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from demo.mock_data import generate_mock_data
from srag.concepts import build_concept_universe
from srag.objective import WeightedCoverageObjective
from srag.selectors import (
    TopKSelector,
    MMRSelector,
    GreedyRelCostSelector,
    SRAGSelector,
)


def run_comparison(candidates, universe, obj, budget, label=""):
    methods = [
        ("Top-k", TopKSelector()),
        ("MMR (λ=0.6)", MMRSelector(universe, lam=0.6)),
        ("Greedy(Rel/Cost)", GreedyRelCostSelector()),
        ("S-RAG (fast)", SRAGSelector(universe, use_lazy=True)),
    ]

    print(f"\n--- Budget B = {budget} {label}---")
    print(f"{'Method':<18} | {'f(S)':>7} | {'Concepts':>8} | {'Budget Used':>14} | {'Docs':>4} | {'Latency':>8}")
    print("-" * 75)

    results = []
    for name, selector in methods:
        t0 = time.perf_counter()
        selected = selector.select(candidates, budget)
        elapsed = (time.perf_counter() - t0) * 1000

        f_val = obj.evaluate(selected)
        covered = obj.covered_concepts(selected)
        total_tokens = sum(d.token_count for d in selected)
        budget_pct = total_tokens / budget * 100

        print(
            f"{name:<18} | {f_val:>7.2f} | {len(covered):>8} | "
            f"{total_tokens:>5}/{budget} ({budget_pct:>5.1f}%) | {len(selected):>4} | "
            f"{elapsed:>6.1f} ms"
        )
        results.append((name, f_val, len(covered), total_tokens, len(selected), elapsed))
    return results


def run_demo():
    TOP_L = 20
    N_CANDIDATES = 200

    print("=" * 75)
    print("  S-RAG Algorithm Reproduction Demo")
    print("  Paper: Optimal Document Selection in RAG via Combinatorial Optimization")
    print("=" * 75)

    query, candidates = generate_mock_data(n_candidates=N_CANDIDATES, seed=42)
    print(f"\nQuery: \"{query.text}\"")
    print(f"Pool size: {len(candidates)} | Concept depth L: {TOP_L}")

    t0 = time.perf_counter()
    universe = build_concept_universe(candidates, top_l=TOP_L)
    concept_time = (time.perf_counter() - t0) * 1000
    print(f"Concept universe: {len(universe.concepts)} unique concepts (built in {concept_time:.1f} ms)")

    obj = WeightedCoverageObjective(universe)

    # Run at multiple budget levels (paper tests B in {1024, 2048, 4096, 8192})
    for budget in [1024, 2048, 4096]:
        run_comparison(candidates, universe, obj, budget)

    # Approximation ratio verification
    print("\n" + "=" * 75)
    print("  Approximation Ratio Verification (brute-force on small subset)")
    print("=" * 75)

    small_candidates = candidates[:10]
    small_universe = build_concept_universe(small_candidates, top_l=10)
    small_obj = WeightedCoverageObjective(small_universe)
    small_budget = 400

    from itertools import combinations
    best_val = 0.0
    best_set = []
    for size in range(1, len(small_candidates) + 1):
        for combo in combinations(small_candidates, size):
            cost = sum(d.token_count for d in combo)
            if cost > small_budget:
                continue
            val = small_obj.evaluate(list(combo))
            if val > best_val:
                best_val = val
                best_set = list(combo)

    small_selector = SRAGSelector(small_universe, use_lazy=False)
    small_selected = small_selector.select(small_candidates, small_budget)
    small_srag_val = small_obj.evaluate(small_selected)

    ratio = small_srag_val / best_val if best_val > 0 else 1.0
    guarantee = 1 - 1 / 2.718281828

    print(f"\n  n={len(small_candidates)}, B={small_budget}")
    print(f"  Brute-force OPT:     {best_val:.3f} ({len(best_set)} docs, {sum(d.token_count for d in best_set)} tokens)")
    print(f"  S-RAG (greedy):      {small_srag_val:.3f} ({len(small_selected)} docs, {sum(d.token_count for d in small_selected)} tokens)")
    print(f"  Approximation ratio: {ratio:.4f}")
    print(f"  Theoretical bound:   >= {guarantee:.4f} (1 - 1/e)")
    print(f"  Result:              {'PASS' if ratio >= guarantee - 0.001 else 'FAIL'}")


if __name__ == "__main__":
    run_demo()
