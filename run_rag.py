"""Unified CLI for end-to-end RAG evaluation (ELI5 wired up first).

Strategy, dataset, models, and budgets are all controlled via flags. Each run
records its launch parameters and results under ckpt/<gen-model>/<run-name>/.
All models are local/open-weight; no commercial LLM API is used.
"""

from __future__ import annotations

import argparse

from rag import config
from rag.device import resolve_device
from rag.recorder import RunRecorder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("eli5",), default="eli5")
    parser.add_argument("--methods", nargs="+", default=list(config.METHODS))
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--budget", type=int, default=config.DEFAULT_BUDGET)
    parser.add_argument("--top-n", type=int, default=config.TOP_N)
    parser.add_argument("--top-l", type=int, default=config.TOP_L)
    parser.add_argument("--mmr-lambda", type=float, default=config.MMR_LAMBDA)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use the small validation models instead of the paper models.",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help='Compute device, e.g. "cuda:0" or "cpu"; falls back to CPU if no GPU.',
    )
    parser.add_argument("--gen-model")
    parser.add_argument("--retriever")
    parser.add_argument("--judge-model")
    parser.add_argument("--bertscore-model")
    parser.add_argument("--corpus-dir", default=str(config.CORPUS_DIR / "eli5"))
    parser.add_argument("--output-dir", default=str(config.CKPT_DIR))
    parser.add_argument("--run-name", default=None)
    parser.add_argument(
        "--judge",
        action="store_true",
        help="Run the open pairwise LLM judge (S-RAG vs baselines, W/T/L).",
    )
    parser.add_argument(
        "--no-test-csv",
        action="store_true",
        help="Skip the per-question test_results.csv deliverable.",
    )
    parser.add_argument(
        "--correctness-threshold",
        type=int,
        default=3,
        help="Absolute grade (1-5) at/above which an answer counts as correct.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    models = config.model_set(args.smoke)
    for key, value in (
        ("retriever", args.retriever),
        ("generator", args.gen_model),
        ("judge", args.judge_model),
        ("bertscore", args.bertscore_model),
    ):
        if value:
            models[key] = value

    import hashlib
    import json

    # Folder identity = model + data settings (NOT the method subset). Runs that
    # differ only in --methods get the same dir and merge their CSVs, so running
    # topk/mmr/greedy/srag separately fills one file instead of duplicating it.
    settings = {
        "dataset": args.dataset,
        "num_samples": args.num_samples,
        "budget": args.budget,
        "top_n": args.top_n,
        "top_l": args.top_l,
        "mmr_lambda": args.mmr_lambda,
        "seed": args.seed,
        "smoke": args.smoke,
        "models": models,
        "corpus_dir": args.corpus_dir,
    }
    sig = hashlib.sha1(
        json.dumps(settings, sort_keys=True, default=str).encode()
    ).hexdigest()[:8]
    run_name = args.run_name or (
        f"{args.dataset}_n{args.num_samples}_b{args.budget}"
        f"_topn{args.top_n}_topl{args.top_l}_mmr{args.mmr_lambda}"
        f"_seed{args.seed}{'_smoke' if args.smoke else ''}_{sig}"
    )
    recorder = RunRecorder(args.output_dir, models["generator"], run_name)

    # Record the union of methods ever run in this settings dir (this run + prior).
    prior_methods: list[str] = []
    config_path = recorder.run_dir / "config.json"
    if config_path.exists():
        try:
            prior_methods = json.loads(config_path.read_text()).get("methods", [])
        except Exception:  # noqa: BLE001 - a corrupt prior config must not abort
            prior_methods = []
    methods_union = list(dict.fromkeys([*prior_methods, *args.methods]))
    # Record the corpus identity (name + passage count) for the run. These are
    # logged only, NOT part of `settings`/the signature, so rebuilding a corpus
    # (which can change the line count) never spuriously forks a new folder.
    from pathlib import Path

    passages_path = Path(args.corpus_dir) / "passages.jsonl"
    corpus_passages = (
        sum(1 for _ in passages_path.open()) if passages_path.exists() else 0
    )
    recorder.save_config(
        {
            **settings,
            "corpus": Path(args.corpus_dir).name,
            "corpus_passages": corpus_passages,
            "methods": methods_union,
            "last_methods": list(args.methods),
        }
    )

    from rag.pipeline import run_eli5

    device = resolve_device(args.device)  # "cuda:0" by default; CPU if no GPU

    rows, summary = run_eli5(
        num_samples=args.num_samples,
        methods=args.methods,
        budget=args.budget,
        model_set=models,
        corpus_dir=args.corpus_dir,
        top_n=args.top_n,
        top_l=args.top_l,
        mmr_lambda=args.mmr_lambda,
        seed=args.seed,
        device=device,
    )
    recorder.save_results(rows)
    recorder.save_summary(summary)

    print(f"\n=== ELI5 results ({len(rows)} rows) -> {recorder.run_dir} ===")
    print(f"{'method':18} {'rougeL':>8} {'BERTScore':>10} {'docs':>6} {'tokens':>8} {'ms':>8}")
    for s in summary:
        print(
            f"{s['method']:18} {s['rougeL']:8.4f} {s['bertscore']:10.4f} "
            f"{s['avg_docs']:6.1f} {s['avg_cost_tokens']:8.1f} {s['avg_latency_ms']:8.1f}"
        )

    need_srag = "srag" in args.methods
    want_judge = args.judge and need_srag
    want_test_csv = not args.no_test_csv  # at least one method is always present

    if want_judge or want_test_csv:
        import gc

        import torch

        from rag.generator import LocalGenerator

        # Free the generator's GPU memory before loading the judge model.
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        baselines = [m for m in args.methods if m != "srag"]
        judge_model = LocalGenerator(models["judge"], device=device)

        if want_judge:
            from rag.judge import judge_run

            wtl = judge_run(judge_model, rows, baselines=baselines)
            recorder.save_judge(wtl)
            print("\n=== Table 2: S-RAG vs baselines (win/tie/loss) ===")
            for r in wtl:
                print(
                    f"{r['comparison']:18} W/T/L = {r['win']}/{r['tie']}/{r['loss']}"
                    f"  ({r['win_pct']}/{r['tie_pct']}/{r['loss_pct']}%)"
                )

        if want_test_csv:
            from rag.export_results import (
                accuracy_summary,
                export_test_results,
                judge_raw_score_fn,
            )

            score_fn = judge_raw_score_fn(judge_model)
            path, test_rows, used = export_test_results(
                rows,
                recorder.run_dir,
                score_fn,
                baselines=baselines,
                threshold=args.correctness_threshold,
            )
            print(f"\n=== Test results CSV ({len(test_rows)} questions) -> {path} ===")
            for r in accuracy_summary(test_rows, used):
                print(
                    f"  {r['column']:32} acc {r['accuracy']:.4f}"
                    f"  ({r['correct']}/{r['total']})"
                )


if __name__ == "__main__":
    main()
