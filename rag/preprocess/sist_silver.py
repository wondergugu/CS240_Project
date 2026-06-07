"""Generate the sist silver QA pool with a local model reading sist passages.

sist ships no human-labeled QA. This builds a one-time pool of (question, short
answer) pairs grounded in the sist corpus passages, written to
data/sist/qa_silver.jsonl. The default generator is the open Prometheus2 model
(user choice); any local instruction model with a `.chat()` method works. All
inference is local — no commercial LLM API.

Quality is enforced after generation: the answer must be short and must appear
(normalized) in the source passage (grounding), questions are de-duplicated, and
generation continues over shuffled passages until the target pool size is met.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

from ..config import DATA_DIR, PAPER_MODELS
from ..corpus import load_passages
from ..metrics import normalize_answer

_SYSTEM = (
    "You write one reading-comprehension QA pair from a passage. The question "
    "must be answerable using ONLY the passage, and the answer must be a short "
    "span (a name, date, number, or short phrase) copied from the passage."
)

_USER_TEMPLATE = (
    "Passage:\n{passage}\n\n"
    "Write exactly two lines and nothing else:\n"
    "Question: <a specific factual question answerable only from the passage>\n"
    "Answer: <the short answer, copied from the passage>"
)

_QUESTION_RE = re.compile(r"question\s*[:：]\s*(.+)", re.IGNORECASE)
_ANSWER_RE = re.compile(r"answer\s*[:：]\s*(.+)", re.IGNORECASE)


def _extract(label_re: re.Pattern[str], text: str) -> str:
    for line in text.splitlines():
        match = label_re.search(line)
        if match:
            return match.group(1).strip().strip("\"'“”《》 ")
    return ""


def parse_qa(
    raw: str,
    source_text: str,
    max_answer_tokens: int = 12,
    max_answer_chars: int = 48,
) -> tuple[str, str] | None:
    """Parse a model reply into a grounded (question, answer), or None if invalid.

    Rejects empty/over-long answers and answers not present (normalized) in the
    source passage, so a hallucinated or generic answer never enters the pool.
    """
    question = _extract(_QUESTION_RE, raw)
    answer = _extract(_ANSWER_RE, raw)
    has_qmark = "?" in question or "？" in question
    if not question or not answer or not has_qmark:
        return None
    if len(answer) > max_answer_chars:
        return None
    norm_answer = normalize_answer(answer)
    if not norm_answer or len(norm_answer.split()) > max_answer_tokens:
        return None
    if norm_answer not in normalize_answer(source_text):
        return None  # grounding check: the answer must come from the passage
    return question, answer


def build_silver_qa(
    corpus_dir: str | Path,
    out_path: str | Path | None = None,
    generator=None,
    gen_model: str | None = None,
    device: str | None = None,
    pool: int = 200,
    seed: int = 42,
    min_chars: int = 200,
    max_new_tokens: int = 64,
) -> tuple[int, Path]:
    """Build (or extend) the sist silver QA pool; return (written, path)."""
    out = Path(out_path) if out_path else Path(DATA_DIR) / "sist" / "qa_silver.jsonl"
    passages = [p for p in load_passages(corpus_dir) if len(p.get("text", "")) >= min_chars]
    random.Random(seed).shuffle(passages)

    if generator is None:
        from ..generator import LocalGenerator

        generator = LocalGenerator(gen_model or PAPER_MODELS["judge"], device=device)

    out.parent.mkdir(parents=True, exist_ok=True)
    seen_questions: set[str] = set()
    written = 0
    with out.open("w", encoding="utf-8") as handle:
        for passage in passages:
            if written >= pool:
                break
            user = _USER_TEMPLATE.format(passage=passage["text"])
            try:
                raw = generator.chat(user, system=_SYSTEM, max_new_tokens=max_new_tokens)
            except Exception as exc:  # noqa: BLE001 - one bad gen must not abort the pool
                print(f"[warn] sist QA gen failed on {passage.get('pid')}: {exc}")
                continue
            parsed = parse_qa(raw, passage["text"])
            if not parsed:
                continue
            question, answer = parsed
            key = normalize_answer(question)
            if key in seen_questions:
                continue
            seen_questions.add(key)
            record = {
                "qid": f"sist_silver_{written}",
                "question": question,
                "answer": answer,
                "references": [answer],
                "source_pid": passage.get("pid", ""),
                "category": passage.get("category", ""),
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
    return written, out


def main(argv=None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Build the sist silver QA pool.")
    parser.add_argument("--corpus-dir", default=str(Path("corpus") / "sist"))
    parser.add_argument("--out", default=None)
    parser.add_argument("--gen-model", default=PAPER_MODELS["judge"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--pool", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    count, path = build_silver_qa(
        args.corpus_dir,
        out_path=args.out,
        gen_model=args.gen_model,
        device=args.device,
        pool=args.pool,
        seed=args.seed,
    )
    print(f"wrote {count} sist silver QA pairs -> {path}")


if __name__ == "__main__":
    main()
