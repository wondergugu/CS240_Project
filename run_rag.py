"""Unified CLI for end-to-end RAG evaluation with a built-in parameter sweep.

Every flag EXCEPT --methods accepts a LIST of values; the script runs the full
Cartesian product (dataset × num_samples × budget × top_n × top_l × mmr_lambda ×
seed), one evaluation per combination, writing each to its own
ckpt/<generator-slug>_<settings-hash>/. The heavy models (retriever, generator,
judge) are loaded ONCE for the whole sweep — generation runs first for every
combo, then the judge is loaded once and scores them all — so a sweep does not
pay the multi-minute model load per combination. --methods is NOT swept: the
whole set is passed to each run and (sharing a settings hash) merges into one
folder. All models are local/open-weight; no commercial LLM API is used.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import itertools
import json
from pathlib import Path

from rag import config, metrics
from rag.device import resolve_device
from rag.pipeline import add_bertscore, build_corpus_index, evaluate, summarize
from rag.preprocess.eli5 import load_eli5_qa
from rag.preprocess.sist import load_sist_qa
from rag.recorder import RunRecorder, _slug


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    # Swept dimensions: each accepts one or more values (Cartesian product).
    parser.add_argument("--dataset", nargs="+", choices=("eli5", "sist"), default=["eli5"])
    parser.add_argument("--num-samples", nargs="+", type=int, default=[50])
    parser.add_argument("--budget", nargs="+", type=int, default=[config.DEFAULT_BUDGET])
    parser.add_argument("--top-n", nargs="+", type=int, default=[config.TOP_N])
    parser.add_argument("--top-l", nargs="+", type=int, default=[config.TOP_L])
    parser.add_argument("--mmr-lambda", nargs="+", type=float, default=[config.MMR_LAMBDA])
    parser.add_argument("--seed", nargs="+", type=int, default=[42])
    # NOT swept: the whole method set is passed to every run.
    parser.add_argument("--methods", nargs="+", default=list(config.METHODS))
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
    parser.add_argument(
        "--corpus-dir",
        default=None,
        help="Override the corpus dir (only valid with a single --dataset); "
        "otherwise corpus/<dataset> is used per dataset.",
    )
    parser.add_argument("--output-dir", default=str(config.CKPT_DIR))
    parser.add_argument(
        "--run-name",
        default=None,
        help="Override the run folder name (only valid for a single combination).",
    )
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


def _dataset_spec(dataset: str) -> dict:
    """Per-dataset QA loader, metric functions, summary keys, and print columns."""
    if dataset == "sist":  # short-answer: EM / token-F1, no BERTScore
        return {
            "loader": load_sist_qa,
            "metric_fns": {"em": metrics.exact_match, "f1": metrics.token_f1},
            "keys": ("em", "f1"),
            "cols": (("em", "EM"), ("f1", "F1")),
            "max_new_tokens": config.MAX_NEW_TOKENS["sist"],
            "bertscore": False,
        }
    return {  # long-form: ROUGE-L + BERTScore
        "loader": load_eli5_qa,
        "metric_fns": {"rougeL": metrics.rouge_l},
        "keys": ("rougeL", "bertscore"),
        "cols": (("rougeL", "rougeL"), ("bertscore", "BERTScore")),
        "max_new_tokens": config.MAX_NEW_TOKENS["eli5"],
        "bertscore": True,
    }


def _resolve_models(args: argparse.Namespace) -> dict[str, str]:
    models = config.model_set(args.smoke)
    for key, value in (
        ("retriever", args.retriever),
        ("generator", args.gen_model),
        ("judge", args.judge_model),
        ("bertscore", args.bertscore_model),
    ):
        if value:
            models[key] = value
    return models


def _record_combo(
    args, models, dataset, corpus_dir, combo, rows, summary, total
) -> RunRecorder:
    """Build the per-combo settings hash, recorder, and write config/results."""
    ns, bd, tn, tl, ml, sd = combo
    settings = {
        "dataset": dataset,
        "num_samples": ns,
        "budget": bd,
        "top_n": tn,
        "top_l": tl,
        "mmr_lambda": ml,
        "seed": sd,
        "smoke": args.smoke,
        "models": models,
        "corpus_dir": corpus_dir,
    }
    sig = hashlib.sha1(
        json.dumps(settings, sort_keys=True, default=str).encode()
    ).hexdigest()[:8]
    single = total == 1
    run_name = (args.run_name if (args.run_name and single) else None) or (
        f"{_slug(models['generator'])}_{sig}"
    )
    recorder = RunRecorder(args.output_dir, models["generator"], run_name, group_by_model=False)

    # Union of methods ever written to this folder (this run + any prior run).
    prior_methods: list[str] = []
    config_path = recorder.run_dir / "config.json"
    if config_path.exists():
        try:
            prior_methods = json.loads(config_path.read_text()).get("methods", [])
        except Exception:  # noqa: BLE001 - a corrupt prior config must not abort
            prior_methods = []
    methods_union = list(dict.fromkeys([*prior_methods, *args.methods]))

    passages_path = Path(corpus_dir) / "passages.jsonl"
    corpus_passages = sum(1 for _ in passages_path.open()) if passages_path.exists() else 0
    recorder.save_config(
        {
            **settings,
            "corpus": Path(corpus_dir).name,
            "corpus_passages": corpus_passages,
            "methods": methods_union,
            "last_methods": list(args.methods),
        }
    )
    recorder.save_results(rows)
    recorder.save_summary(summary)
    return recorder


def main() -> None:
    args = parse_args()
    models = _resolve_models(args)
    device = resolve_device(args.device)  # "cuda:0" by default; CPU if no GPU
    baselines = [m for m in args.methods if m != "srag"]

    sweep = list(
        itertools.product(
            args.num_samples, args.budget, args.top_n, args.top_l, args.mmr_lambda, args.seed
        )
    )
    total = len(args.dataset) * len(sweep)
    print(
        f"[run_rag] sweep = {total} run(s) | datasets={args.dataset} "
        f"methods(NOT swept)={args.methods} smoke={args.smoke} device={device}"
    )

    # ---- Phase A: load retriever + generator ONCE, generate over every combo ----
    from rag.generator import LocalGenerator
    from rag.retriever import BGERetriever

    retriever = BGERetriever(models["retriever"], device=device)
    generator = LocalGenerator(models["generator"], device=device)

    completed: list[dict] = []
    done = 0
    for dataset in args.dataset:
        corpus_dir = (
            args.corpus_dir
            if (args.corpus_dir and len(args.dataset) == 1)
            else f"corpus/{dataset}"
        )
        spec = _dataset_spec(dataset)
        index = build_corpus_index(retriever, corpus_dir, device, cache_key=models["retriever"])

        for combo in sweep:
            ns, bd, tn, tl, ml, sd = combo
            done += 1
            print(
                f"[run_rag] === run {done}/{total}: dataset={dataset} n={ns} budget={bd} "
                f"top_n={tn} top_l={tl} mmr={ml} seed={sd} ==="
            )
            rows = evaluate(
                examples=spec["loader"](num_samples=ns, seed=sd),
                retriever=retriever,
                index=index,
                generator=generator,
                methods=args.methods,
                budget=bd,
                top_n=tn,
                top_l=tl,
                mmr_lambda=ml,
                max_new_tokens=spec["max_new_tokens"],
                metric_fns=spec["metric_fns"],
            )
            if spec["bertscore"]:
                add_bertscore(rows, models)
            summary = summarize(rows, args.methods, metric_keys=spec["keys"])
            recorder = _record_combo(
                args, models, dataset, corpus_dir, combo, rows, summary, total
            )

            print(f"  -> {recorder.run_dir} ({len(rows)} rows)")
            header = f"{'method':18}" + "".join(f"{label:>10}" for _, label in spec["cols"])
            print(header + f"{'docs':>7}{'tokens':>9}{'ms':>9}")
            for s in summary:
                cells = "".join(f"{s.get(k, 0.0):10.4f}" for k, _ in spec["cols"])
                print(
                    f"{s['method']:18}{cells}"
                    f"{s['avg_docs']:7.1f}{s['avg_cost_tokens']:9.1f}{s['avg_latency_ms']:9.1f}"
                )
            completed.append({"recorder": recorder, "rows": rows, "dataset": dataset})

    # Free the generator's GPU memory before loading the judge.
    del generator
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001 - cache cleanup is best-effort
        pass

    # ---- Phase B: load the judge ONCE, score test_results.csv (+ W/T/L) per combo ----
    want_judge = args.judge and "srag" in args.methods
    want_test_csv = not args.no_test_csv
    if not (completed and (want_judge or want_test_csv)):
        return

    from rag.export_results import accuracy_summary, export_test_results, judge_raw_score_fn

    judge_model = LocalGenerator(models["judge"], device=device)
    score_fn = judge_raw_score_fn(judge_model)

    for entry in completed:
        recorder, rows = entry["recorder"], entry["rows"]
        if want_judge:
            from rag.judge import judge_run

            wtl = judge_run(judge_model, rows, baselines=baselines)
            recorder.save_judge(wtl)
            print(f"\n=== Table 2 (W/T/L) -> {recorder.run_dir} ===")
            for r in wtl:
                print(
                    f"{r['comparison']:18} W/T/L = {r['win']}/{r['tie']}/{r['loss']}"
                    f"  ({r['win_pct']}/{r['tie_pct']}/{r['loss_pct']}%)"
                )
        if want_test_csv:
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
