"""Compare document selectors on the paper QA datasets."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from itertools import combinations
from math import e
from pathlib import Path
import statistics
import time

from demo.dataset_loader import (
    DEFAULT_DATA_ROOT,
    PAPER_SAMPLE_SIZE,
    SUPPORTED_DATASETS,
    DatasetNotFoundError,
    PaperDatasetExample,
    load_paper_dataset,
)
from srag.concepts import build_concept_universe
from srag.objective import WeightedCoverageObjective
from srag.selectors import (
    GreedyRelCostSelector,
    MMRSelector,
    SRAGSelector,
    TopKSelector,
)
from srag.types import ConceptUniverse
from srag.utils import total_cost

DEFAULT_BUDGETS = (1024, 2048, 4096)


@dataclass(slots=True)
class PreparedExample:
    source: PaperDatasetExample
    universe: ConceptUniverse
    objective: WeightedCoverageObjective


@dataclass(slots=True)
class MethodStats:
    values: list[float]
    concepts: list[int]
    costs: list[int]
    docs: list[int]
    recalls: list[float]
    latencies_ms: list[float]

    @classmethod
    def empty(cls) -> "MethodStats":
        return cls([], [], [], [], [], [])

    def add(
        self,
        value: float,
        concepts: int,
        cost: int,
        docs: int,
        recall: float | None,
        latency_ms: float,
    ) -> None:
        self.values.append(value)
        self.concepts.append(concepts)
        self.costs.append(cost)
        self.docs.append(docs)
        if recall is not None:
            self.recalls.append(recall)
        self.latencies_ms.append(latency_ms)


def _mean(values: list[float] | list[int]) -> float:
    return statistics.fmean(values) if values else 0.0


def prepare_examples(
    examples: list[PaperDatasetExample], top_l: int
) -> list[PreparedExample]:
    prepared: list[PreparedExample] = []
    for example in examples:
        universe = build_concept_universe(
            example.candidates, top_l=min(top_l, len(example.candidates))
        )
        prepared.append(
            PreparedExample(
                source=example,
                universe=universe,
                objective=WeightedCoverageObjective(universe),
            )
        )
    return prepared


def run_comparison(prepared: list[PreparedExample], budget: int) -> None:
    stats = {
        "Top-k": MethodStats.empty(),
        "MMR (concept proxy)": MethodStats.empty(),
        "Greedy (Rel/Cost)": MethodStats.empty(),
        "S-RAG (fast)": MethodStats.empty(),
    }

    for item in prepared:
        methods = [
            ("Top-k", TopKSelector()),
            ("MMR (concept proxy)", MMRSelector(item.universe, lam=0.6)),
            ("Greedy (Rel/Cost)", GreedyRelCostSelector()),
            ("S-RAG (fast)", SRAGSelector(item.universe, use_lazy=True)),
        ]
        for name, selector in methods:
            started = time.perf_counter()
            selected = selector.select(item.source.candidates, budget)
            elapsed_ms = (time.perf_counter() - started) * 1000

            selected_ids = {doc.doc_id for doc in selected}
            gold_ids = item.source.gold_doc_ids
            recall = None
            if gold_ids:
                recall = len(selected_ids & gold_ids) / len(gold_ids)

            stats[name].add(
                value=item.objective.evaluate(selected),
                concepts=len(item.objective.covered_concepts(selected)),
                cost=total_cost(selected),
                docs=len(selected),
                recall=recall,
                latency_ms=elapsed_ms,
            )

    print(f"\n--- Budget B = {budget} ---")
    print(
        f"{'Method':<20} | {'f(S)':>8} | {'Concepts':>8} | "
        f"{'RefRecall':>9} | {'Budget':>9} | {'Docs':>6} | {'Latency':>8}"
    )
    print("-" * 86)

    for name, method_stats in stats.items():
        recall_text = "n/a"
        if method_stats.recalls:
            recall_text = f"{_mean(method_stats.recalls):.3f}"
        print(
            f"{name:<20} | {_mean(method_stats.values):>8.2f} | "
            f"{_mean(method_stats.concepts):>8.1f} | {recall_text:>9} | "
            f"{_mean(method_stats.costs):>9.1f} | "
            f"{_mean(method_stats.docs):>6.1f} | "
            f"{_mean(method_stats.latencies_ms):>6.1f} ms"
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


def run_small_instance_check(prepared: list[PreparedExample]) -> None:
    if not prepared:
        return

    source = prepared[0].source
    small_candidates = source.candidates[:10]
    if len(small_candidates) < 3:
        print("\nSmall-instance check skipped: fewer than three candidates.")
        return

    universe = build_concept_universe(
        small_candidates, top_l=min(10, len(small_candidates))
    )
    objective = WeightedCoverageObjective(universe)
    budget = max(1, min(400, sum(doc.token_count for doc in small_candidates)))

    optimum, optimum_value = brute_force_optimum(
        small_candidates, objective, budget
    )
    selector = SRAGSelector(universe, use_lazy=False)
    fast = selector.select(small_candidates, budget)
    theoretical = selector.select_theoretical(small_candidates, budget)
    theoretical_value = objective.evaluate(theoretical)
    ratio = theoretical_value / optimum_value if optimum_value else 1.0
    guarantee = 1 - 1 / e

    print("\n" + "=" * 86)
    print("  Small-Instance Check: Fast Variant vs. Algorithm 1")
    print("=" * 86)
    print(
        f"\n  dataset={source.dataset}, n={len(small_candidates)}, B={budget}\n"
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


def run_dataset(
    dataset: str,
    data_root: Path,
    sample_size: int,
    seed: int,
    top_l: int,
    budgets: tuple[int, ...],
    max_candidates: int,
) -> bool:
    try:
        examples = load_paper_dataset(
            dataset=dataset,
            data_root=data_root,
            sample_size=sample_size,
            seed=seed,
            max_candidates=max_candidates,
        )
    except DatasetNotFoundError as exc:
        print(f"\n[{dataset}] {exc}")
        return False

    if not examples:
        print(f"\n[{dataset}] No usable examples found in local Parquet files.")
        return False

    print("\n" + "=" * 86)
    print(f"  Dataset: {dataset}")
    print("=" * 86)
    print(
        f"Requested samples: {sample_size} | Loaded usable samples: {len(examples)} | "
        f"Candidate cap/query: {max_candidates} | Concept depth L: {top_l}"
    )
    print(f"Example query: {examples[0].query.text[:180]}")

    started = time.perf_counter()
    prepared = prepare_examples(examples, top_l=top_l)
    elapsed = time.perf_counter() - started
    avg_candidates = _mean([len(example.candidates) for example in examples])
    avg_gold = _mean([len(example.gold_doc_ids) for example in examples])
    print(
        f"Prepared concept universes in {elapsed:.1f}s | "
        f"Avg candidates/query: {avg_candidates:.1f} | Avg reference docs/query: {avg_gold:.1f}"
    )

    for budget in budgets:
        run_comparison(prepared, budget)

    run_small_instance_check(prepared)
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run S-RAG selectors on paper QA datasets."
    )
    parser.add_argument(
        "--dataset",
        choices=("all",) + SUPPORTED_DATASETS,
        default="all",
        help="Paper dataset to evaluate. Default: all.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Directory containing data/eli5 and data/hotpotqa.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=PAPER_SAMPLE_SIZE,
        help="Number of examples sampled per dataset. Default: 500.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-l", type=int, default=20)
    parser.add_argument("--max-candidates", type=int, default=20)
    parser.add_argument(
        "--budgets",
        type=int,
        nargs="+",
        default=list(DEFAULT_BUDGETS),
        help="Token budgets to evaluate.",
    )
    return parser.parse_args()


def run_demo() -> None:
    args = parse_args()
    datasets = SUPPORTED_DATASETS if args.dataset == "all" else (args.dataset,)

    print("=" * 86)
    print("  S-RAG Document Selection Demo on Paper QA Datasets")
    print("  Default evaluation samples 500 examples per dataset.")
    print("  Metrics are selection-side proxies; no LLM answer EM/F1 is computed.")
    print("=" * 86)

    completed = 0
    for dataset in datasets:
        if run_dataset(
            dataset=dataset,
            data_root=args.data_root,
            sample_size=args.sample_size,
            seed=args.seed,
            top_l=args.top_l,
            budgets=tuple(args.budgets),
            max_candidates=args.max_candidates,
        ):
            completed += 1

    if completed == 0:
        raise SystemExit(
            "No dataset evaluation ran. Download the paper datasets using README.md first."
        )


if __name__ == "__main__":
    run_demo()
