"""End-to-end RAG evaluation pipeline (ELI5 long-form path).

For each question: retrieve top-N passages from the independent corpus, run each
selection strategy under the token budget, generate an answer with the local
model, and score it. ELI5 is long-form, so the per-answer metrics are ROUGE-L
and BERTScore.
"""

from __future__ import annotations

import statistics
import time
from collections.abc import Sequence

from . import config, metrics
from .corpus import TorchCorpusIndex, load_passages
from .device import resolve_device
from .generator import LocalGenerator
from .preprocess.eli5 import load_eli5_qa
from .preprocess.sist import load_sist_qa
from .retriever import BGERetriever
from .selection import Candidate, build_documents, select


def _passage_text(passage: dict) -> str:
    title = passage.get("title") or ""
    return f"{title}. {passage['text']}" if title else passage["text"]


def _make_similarity(retriever: BGERetriever, documents):
    import numpy as np

    texts = [d.text for d in documents]
    embeddings = retriever.encode_passages(texts)
    by_id = {doc.doc_id: emb for doc, emb in zip(documents, embeddings)}

    def similarity(doc_a, doc_b) -> float:
        a, b = by_id.get(doc_a.doc_id), by_id.get(doc_b.doc_id)
        return float(np.dot(a, b)) if a is not None and b is not None else 0.0

    return similarity


def run_eli5(
    *,
    num_samples: int,
    methods: Sequence[str],
    budget: int,
    model_set: dict[str, str],
    corpus_dir: str,
    top_n: int = 200,
    top_l: int = 20,
    mmr_lambda: float = 0.5,
    seed: int = 42,
    device: str | None = None,
) -> tuple[list[dict], list[dict]]:
    """Run the ELI5 evaluation and return (per-instance rows, summary rows)."""
    device = resolve_device(device)  # honor "cuda:0" etc., fall back to CPU
    retriever = BGERetriever(model_set["retriever"], device=device)
    index = TorchCorpusIndex(
        load_passages(corpus_dir),
        retriever,
        device=device,
        cache_dir=corpus_dir,
        cache_key=model_set["retriever"],
    )
    generator = LocalGenerator(model_set["generator"], device=device)
    examples = load_eli5_qa(num_samples=num_samples, seed=seed)
    max_new_tokens = config.MAX_NEW_TOKENS["eli5"]

    rows: list[dict] = []
    for example in examples:
        retrieved = index.retrieve(example.question, top_n=top_n)
        if not retrieved:
            continue
        candidates = [
            Candidate(p["pid"], _passage_text(p), (p["score"] + 1.0) / 2.0)
            for p in retrieved
        ]
        documents = build_documents(candidates, generator.count_tokens)
        similarity = (
            _make_similarity(retriever, documents) if "mmr" in methods else None
        )
        reference = max(example.references, key=len)

        for method in methods:
            started = time.perf_counter()
            selected = select(
                method,
                documents,
                budget,
                top_l=top_l,
                similarity=similarity,
                mmr_lambda=mmr_lambda,
            )
            answer = generator.generate(
                example.question, [d.text for d in selected], max_new_tokens
            )
            rows.append(
                {
                    "qid": example.qid,
                    "question": example.question,
                    "method": method,
                    "prediction": answer,
                    "reference": reference,
                    "rougeL": round(metrics.rouge_l(answer, list(example.references)), 4),
                    "n_docs": len(selected),
                    "cost_tokens": sum(d.token_count for d in selected),
                    "latency_ms": round((time.perf_counter() - started) * 1000, 1),
                }
            )

    if rows:
        try:
            f1 = metrics.bertscore_f1(
                [r["prediction"] for r in rows],
                [r["reference"] for r in rows],
                model_type=model_set["bertscore"],
                num_layers=config.BERTSCORE_LAYERS.get(model_set["bertscore"]),
            )
            for row, score in zip(rows, f1):
                row["bertscore"] = round(score, 4)
        except Exception as exc:  # noqa: BLE001 - metric must not abort the run
            print(f"[warn] BERTScore skipped ({type(exc).__name__}: {exc})")

    return rows, _summarize(rows, methods)


def run_sist(
    *,
    num_samples: int,
    methods: Sequence[str],
    budget: int,
    model_set: dict[str, str],
    corpus_dir: str,
    top_n: int = 200,
    top_l: int = 20,
    mmr_lambda: float = 0.5,
    seed: int = 42,
    device: str | None = None,
) -> tuple[list[dict], list[dict]]:
    """Run the sist evaluation and return (per-instance rows, summary rows).

    sist is a short-answer dataset, so per-answer quality is Exact Match and
    token F1 (no BERTScore). Retrieval is against the local sist corpus, whose
    passage embeddings are cached on disk (see TorchCorpusIndex).
    """
    device = resolve_device(device)
    retriever = BGERetriever(model_set["retriever"], device=device)
    index = TorchCorpusIndex(
        load_passages(corpus_dir),
        retriever,
        device=device,
        cache_dir=corpus_dir,
        cache_key=model_set["retriever"],
    )
    generator = LocalGenerator(model_set["generator"], device=device)
    examples = load_sist_qa(num_samples=num_samples, seed=seed)
    max_new_tokens = config.MAX_NEW_TOKENS["sist"]

    rows: list[dict] = []
    for example in examples:
        retrieved = index.retrieve(example.question, top_n=top_n)
        if not retrieved:
            continue
        candidates = [
            Candidate(p["pid"], _passage_text(p), (p["score"] + 1.0) / 2.0)
            for p in retrieved
        ]
        documents = build_documents(candidates, generator.count_tokens)
        similarity = (
            _make_similarity(retriever, documents) if "mmr" in methods else None
        )
        reference = max(example.references, key=len)

        for method in methods:
            started = time.perf_counter()
            selected = select(
                method,
                documents,
                budget,
                top_l=top_l,
                similarity=similarity,
                mmr_lambda=mmr_lambda,
            )
            answer = generator.generate(
                example.question, [d.text for d in selected], max_new_tokens
            )
            rows.append(
                {
                    "qid": example.qid,
                    "question": example.question,
                    "method": method,
                    "prediction": answer,
                    "reference": reference,
                    "em": round(metrics.exact_match(answer, list(example.references)), 4),
                    "f1": round(metrics.token_f1(answer, list(example.references)), 4),
                    "n_docs": len(selected),
                    "cost_tokens": sum(d.token_count for d in selected),
                    "latency_ms": round((time.perf_counter() - started) * 1000, 1),
                }
            )

    return rows, _summarize(rows, methods, metric_keys=("em", "f1"))


def _summarize(
    rows: list[dict],
    methods: Sequence[str],
    metric_keys: Sequence[str] = ("rougeL", "bertscore"),
) -> list[dict]:
    summary: list[dict] = []
    for method in methods:
        method_rows = [r for r in rows if r["method"] == method]
        if not method_rows:
            continue

        def mean(key: str) -> float:
            return statistics.fmean(r.get(key, 0.0) for r in method_rows)

        entry = {"method": method, "n": len(method_rows)}
        for key in metric_keys:
            entry[key] = round(mean(key), 4)
        entry["avg_docs"] = round(mean("n_docs"), 2)
        entry["avg_cost_tokens"] = round(mean("cost_tokens"), 1)
        entry["avg_latency_ms"] = round(mean("latency_ms"), 1)
        summary.append(entry)
    return summary
