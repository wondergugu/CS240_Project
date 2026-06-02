"""Synthetic comparison of practical selectors and theoretical S-RAG."""

from itertools import combinations
from math import e
import time

from demo.mock_data import generate_mock_data
from srag.concepts import build_concept_universe
from srag.objective import WeightedCoverageObjective
from srag.selectors import (
    GreedyRelCostSelector,
    MMRSelector,
    SRAGSelector,
    TopKSelector,
)
from srag.utils import total_cost


def run_comparison(candidates, universe, objective, budget):
    methods = [
        ("Top-k", TopKSelector()),
        ("MMR (concept proxy)", MMRSelector(universe, lam=0.6)),
        ("Greedy (Rel/Cost)", GreedyRelCostSelector()),
        ("S-RAG (fast)", SRAGSelector(universe, use_lazy=True)),
    ]

    print(f"\n--- Budget B = {budget} ---")
    print(
        f"{'Method':<20} | {'f(S)':>7} | {'Concepts':>8} | "
        f"{'Budget Used':>14} | {'Docs':>4} | {'Latency':>8}"
    )
    print("-" * 79)

    for name, selector in methods:
        started = time.perf_counter()
        selected = selector.select(candidates, budget)
        elapsed_ms = (time.perf_counter() - started) * 1000

        value = objective.evaluate(selected)
        covered = objective.covered_concepts(selected)
        cost = total_cost(selected)
        budget_pct = cost / budget * 100

        print(
            f"{name:<20} | {value:>7.2f} | {len(covered):>8} | "
            f"{cost:>5}/{budget} ({budget_pct:>5.1f}%) | "
            f"{len(selected):>4} | {elapsed_ms:>6.1f} ms"
        )


def brute_force_optimum(candidates, objective, budget):
    best = []
    best_value = 0.0
    for size in range(len(candidates) + 1):
        for candidate in combinations(candidates, size):
            if total_cost(candidate) > budget:
                continue
            value = objective.evaluate(candidate)
            if value > best_value:
                best_value = value
                best = list(candidate)
    return best, best_value


def run_small_instance_check(candidates):
    small_candidates = candidates[:10]
    universe = build_concept_universe(small_candidates, top_l=10)
    objective = WeightedCoverageObjective(universe)
    budget = 400

    optimum, optimum_value = brute_force_optimum(
        small_candidates, objective, budget
    )
    selector = SRAGSelector(universe, use_lazy=False)
    fast = selector.select(small_candidates, budget)
    theoretical = selector.select_theoretical(small_candidates, budget)
    theoretical_value = objective.evaluate(theoretical)
    ratio = theoretical_value / optimum_value if optimum_value else 1.0
    guarantee = 1 - 1 / e

    print("\n" + "=" * 79)
    print("  Small-Instance Check: Fast Variant vs. Algorithm 1")
    print("=" * 79)
    print(
        f"\n  n={len(small_candidates)}, B={budget}\n"
        f"  Brute-force OPT:       {optimum_value:.3f} "
        f"({len(optimum)} docs, {total_cost(optimum)} tokens)\n"
        f"  S-RAG (fast):          {objective.evaluate(fast):.3f} "
        f"({len(fast)} docs, {total_cost(fast)} tokens)\n"
        f"  S-RAG (Algorithm 1):   {theoretical_value:.3f} "
        f"({len(theoretical)} docs, {total_cost(theoretical)} tokens)\n"
        f"  Observed ratio to OPT: {ratio:.4f}\n"
        f"  Theorem reference:     >= {guarantee:.4f} (1 - 1/e)\n"
        f"  Observed bound check:  "
        f"{'PASS' if ratio >= guarantee - 1e-12 else 'FAIL'}"
    )


def run_demo():
    top_l = 20
    n_candidates = 200

    print("=" * 79)
    print("  S-RAG Document Selection Demo (Synthetic Candidate Pool)")
    print("  Practical comparison uses fast density-greedy selection.")
    print("  MMR uses concept Jaccard because this demo has no embeddings.")
    print("=" * 79)

    query, candidates = generate_mock_data(
        n_candidates=n_candidates, seed=42
    )
    print(f'\nQuery: "{query.text}"')
    print(f"Pool size: {len(candidates)} | Concept depth L: {top_l}")

    started = time.perf_counter()
    universe = build_concept_universe(candidates, top_l=top_l)
    elapsed_ms = (time.perf_counter() - started) * 1000
    print(
        f"Concept universe: {len(universe.concepts)} unique concepts "
        f"(built in {elapsed_ms:.1f} ms)"
    )

    objective = WeightedCoverageObjective(universe)
    for budget in (1024, 2048, 4096):
        run_comparison(candidates, universe, objective, budget)

    run_small_instance_check(candidates)


if __name__ == "__main__":
    run_demo()
