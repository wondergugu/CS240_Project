"""Interactive S-RAG inference over the local sist knowledge base.

Loads the paper models ONCE (bge retriever + cached corpus index + Llama-3
generator) and answers a free-form user question end to end, reusing the exact
retrieve -> select -> generate path of ``rag.pipeline.evaluate`` so the demo and
the offline evaluation agree. The selection strategy is fixed to S-RAG (the
paper's submodular method); the answer is the local generator's short extractive
span over the selected passages. All models are open-weight and run locally — no
commercial LLM API is ever called.

CLI / REPL (no streamlit needed; good for a quick cuda:0 check):

    python3 -m rag.infer "信息科学与技术学院有哪些研究中心?"   # one-shot
    python3 -m rag.infer                                        # interactive REPL
    # options: --device cuda:0 --top-n 200 --budget 4096 --max-new-tokens 32 --long
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from . import config
from .corpus import load_passages
from .device import resolve_device
from .generator import LocalGenerator
from .pipeline import _passage_text, build_corpus_index
from .retriever import BGERetriever
from .selection import Candidate, build_documents, select

# Interactive answers default to a longer cap than the eval pipeline's sist
# setting (config.MAX_NEW_TOKENS["sist"]=32, kept tight there so Exact Match
# fires). The demo favours fuller answers (e.g. list questions) over strict span
# extraction, so it decouples its own default here.
DEFAULT_MAX_NEW_TOKENS = 256


@dataclass
class Evidence:
    """One S-RAG-selected source passage, with provenance for display."""

    pid: str
    title: str
    url: str
    category: str
    score: float  # retriever cosine relevance rel(d, q)
    tokens: int   # generator-token cost c(d)
    text: str


@dataclass
class Answer:
    """Result of one question: the generated answer plus its evidence + stats."""

    question: str
    answer: str
    method: str = "srag"
    evidence: list[Evidence] = field(default_factory=list)
    n_candidates: int = 0
    n_selected: int = 0
    cost_tokens: int = 0
    budget: int = 0
    latency_ms: float = 0.0


class SistRAG:
    """End-to-end S-RAG question answering over the sist corpus (load once)."""

    def __init__(
        self,
        *,
        device: str | None = "cuda:0",
        smoke: bool = False,
        corpus_dir: str | None = None,
        budget: int = config.DEFAULT_BUDGET,
        top_n: int = config.TOP_N,
        top_l: int = config.TOP_L,
        max_new_tokens: int | None = None,
    ):
        self.device = resolve_device(device)
        self.models = config.model_set(smoke)
        self.corpus_dir = str(corpus_dir or (config.CORPUS_DIR / "sist"))
        self.budget = budget
        self.top_n = top_n
        self.top_l = top_l
        self.max_new_tokens = max_new_tokens or DEFAULT_MAX_NEW_TOKENS

        self.retriever = BGERetriever(self.models["retriever"], device=self.device)
        # Keep passage metadata (title/url/category) for provenance display.
        self._meta = {p["pid"]: p for p in load_passages(self.corpus_dir)}
        # Reuses the on-disk embedding cache under corpus/sist when present.
        self.index = build_corpus_index(
            self.retriever,
            self.corpus_dir,
            self.device,
            cache_key=self.models["retriever"],
        )
        self.generator = LocalGenerator(self.models["generator"], device=self.device)

    def answer(
        self,
        question: str,
        *,
        method: str = "srag",
        top_n: int | None = None,
        budget: int | None = None,
        max_new_tokens: int | None = None,
        short: bool = True,
    ) -> Answer:
        """Retrieve -> S-RAG select under the token budget -> generate an answer.

        Mirrors one inner iteration of ``rag.pipeline.evaluate`` so the demo and
        the offline numbers are produced by the same code path.
        """
        started = time.perf_counter()
        top_n = top_n or self.top_n
        budget = budget or self.budget
        max_new_tokens = max_new_tokens or self.max_new_tokens

        retrieved = self.index.retrieve(question, top_n=top_n)
        if not retrieved:
            return Answer(
                question=question,
                answer="(检索为空:语料中未找到相关段落)",
                method=method,
                budget=budget,
                latency_ms=round((time.perf_counter() - started) * 1000, 1),
            )

        candidates = [
            Candidate(p["pid"], _passage_text(p), (p["score"] + 1.0) / 2.0)
            for p in retrieved
        ]
        documents = build_documents(candidates, self.generator.count_tokens)
        # S-RAG needs no pairwise similarity (that is MMR's proxy), so none built.
        selected = select(method, documents, budget, top_l=self.top_l)
        text = self.generator.generate(
            question, [d.text for d in selected], max_new_tokens, short=short
        )

        score_by_id = {p["pid"]: p["score"] for p in retrieved}
        evidence = [
            Evidence(
                pid=d.doc_id,
                title=self._meta.get(d.doc_id, {}).get("title", ""),
                url=self._meta.get(d.doc_id, {}).get("url", ""),
                category=self._meta.get(d.doc_id, {}).get("category", ""),
                score=float(score_by_id.get(d.doc_id, 0.0)),
                tokens=d.token_count,
                text=self._meta.get(d.doc_id, {}).get("text", d.text),
            )
            for d in selected
        ]
        return Answer(
            question=question,
            answer=text,
            method=method,
            evidence=evidence,
            n_candidates=len(documents),
            n_selected=len(selected),
            cost_tokens=sum(d.token_count for d in selected),
            budget=budget,
            latency_ms=round((time.perf_counter() - started) * 1000, 1),
        )


def _print_answer(ans: Answer, show: int = 3) -> None:
    print(f"\nQ: {ans.question}")
    print(f"A: {ans.answer}")
    print(
        f"   [method={ans.method}  selected={ans.n_selected}/{ans.n_candidates}"
        f"  ctx_tokens={ans.cost_tokens}/{ans.budget}"
        f"  {ans.latency_ms / 1000:.2f}s]"
    )
    for i, e in enumerate(ans.evidence[:show], 1):
        head = e.title or e.pid
        url = f"  {e.url}" if e.url else ""
        print(f"   [{i}] {head}  (score={e.score:.3f}, {e.tokens} tok){url}")


def main(argv=None) -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Interactive S-RAG QA over the local sist corpus."
    )
    parser.add_argument("question", nargs="?", help="one-shot question; omit for REPL")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--smoke", action="store_true", help="small validation models")
    parser.add_argument("--top-n", type=int, default=config.TOP_N)
    parser.add_argument("--budget", type=int, default=config.DEFAULT_BUDGET)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument(
        "--long", action="store_true", help="synthesis (long) answer, not a short span"
    )
    parser.add_argument("--show", type=int, default=3, help="evidence passages to print")
    args = parser.parse_args(argv)

    print(f"[infer] loading models on {args.device} (one-time)…", flush=True)
    rag = SistRAG(
        device=args.device,
        smoke=args.smoke,
        top_n=args.top_n,
        budget=args.budget,
        max_new_tokens=args.max_new_tokens,
    )
    short = not args.long

    def ask(q: str) -> None:
        _print_answer(rag.answer(q, short=short), show=args.show)

    if args.question:
        ask(args.question)
        return
    print("[infer] interactive mode — type a question, blank line or Ctrl-D to quit.")
    while True:
        try:
            q = input("\n> ").strip()
        except EOFError:
            break
        if not q:
            break
        ask(q)


if __name__ == "__main__":
    main()
