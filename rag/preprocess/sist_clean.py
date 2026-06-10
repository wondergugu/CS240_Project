"""Clean the sist silver QA pool: drop unrecoverable garbage, keep CJK answers.

The silver pool (rag/preprocess/sist_silver.py) is already gated at generation
time (answer <=12 tokens / <=64 chars + grounded in the source passage), so it is
NOT full of noise. A scan of the 200-record pool found only a handful of truly
broken answers (PDF glyph/CID artifacts, raw formulas) versus ~47 valid Chinese
(CJK) answers whose only problem is whitespace tokenization — handled by the
char-level tokenizer in rag/metrics.py, NOT by deletion.

So this cleaner removes only the genuine garbage and (by default) re-verifies
grounding, while KEEPING CJK answers. It writes data/sist/qa_silver_clean.jsonl
and never touches the original qa_silver.jsonl. All local; no network, no new
dependency (only re/json/unicode ranges).

CLI:
    python3 -m rag.preprocess.sist_clean \
        --in data/sist/qa_silver.jsonl --out data/sist/qa_silver_clean.jsonl \
        --corpus-dir corpus/sist            # grounding re-check (default on)
    # options: --drop-cjk (exclude Chinese answers), --no-grounding
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from ..config import DATA_DIR
from ..corpus import load_passages
from ..metrics import _is_cjk, normalize_answer

# Rule 1: PDF extraction artifacts that can never match a clean gold answer.
_GLYPH_RE = re.compile(r"/uni[0-9a-fA-F]{4,6}|\(cid:\d+\)|/glyph\d+|�")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
# Rule 2: math/formula symbols — a span full of these is a copied equation.
_MATH_RE = re.compile(r"[∑∏∫√≤≥±‖→←↔∈∉⊆⊇∩∪≈≠∞∂∇·×÷]")
_FUNCTION_TAIL = re.compile(r"\b(of|a|an|the|and|in|to|for|with|on|at|by)$", re.I)
_STOPWORDS = {"the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "is", "are"}


def _answer(record: dict) -> str:
    refs = record.get("references")
    if isinstance(refs, list) and refs:
        return str(refs[0])
    return str(record.get("answer", ""))


def _has_cjk(text: str) -> bool:
    return any(_is_cjk(ch) for ch in text)


def _symbol_ratio(text: str) -> float:
    """Fraction of non-space chars that are neither alphanumeric nor CJK."""
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return 0.0
    junk = sum(1 for c in chars if not (c.isalnum() or _is_cjk(c)))
    return junk / len(chars)


def classify(
    record: dict, *, drop_cjk: bool = False, grounding: dict | None = None
) -> tuple[str, str]:
    """Return (status, reason); status in {keep, cjk_keep, flag, drop}."""
    answer = _answer(record).strip()
    question = str(record.get("question", "")).strip()

    if not answer:
        return "drop", "empty_answer"
    if _GLYPH_RE.search(answer) or _CONTROL_RE.search(answer):
        return "drop", "glyph_cid_garbage"
    if _MATH_RE.search(answer) or _symbol_ratio(answer) > 0.5:
        return "drop", "formula_symbols"

    norm = normalize_answer(answer)
    if not norm:
        return "drop", "empty_after_normalize"
    if norm == normalize_answer(question):
        return "drop", "answer_equals_question"
    if all(tok in _STOPWORDS for tok in norm.split()):
        return "drop", "all_stopwords"

    cjk = _has_cjk(answer)
    if not cjk and len(answer.split()) > 12:
        return "drop", "too_long"

    # Grounding re-verification (default on): the answer must still appear,
    # normalized, in its source passage. CJK answers are checked the same way.
    if grounding is not None:
        pid = str(record.get("source_pid", ""))
        source = grounding.get(pid)
        if source is not None and norm not in source:
            return "drop", "grounding_failed"

    if cjk:
        return ("drop", "cjk_dropped") if drop_cjk else ("cjk_keep", "cjk")
    if _FUNCTION_TAIL.search(norm) or (answer[:1].islower() and len(answer.split()) > 3):
        return "flag", "fragment_low_confidence"
    return "keep", "ok"


def _grounding_index(corpus_dir: str | Path) -> dict[str, str]:
    """pid -> normalized passage text, for the grounding re-check."""
    index: dict[str, str] = {}
    for passage in load_passages(corpus_dir):
        pid = str(passage.get("pid", ""))
        if pid:
            index[pid] = normalize_answer(passage.get("text", ""))
    return index


def clean_pool(
    in_path: str | Path,
    out_path: str | Path,
    *,
    drop_cjk: bool = False,
    verify_grounding: bool = True,
    corpus_dir: str | Path = "corpus/sist",
) -> dict:
    """Filter the silver pool into a clean pool; return per-status/reason stats.

    The original `in_path` is read-only; only `out_path` is written. Records with
    status keep/cjk_keep/flag are written; only `drop` is excluded.
    """
    in_path, out_path = Path(in_path), Path(out_path)
    grounding = _grounding_index(corpus_dir) if verify_grounding else None

    status_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    kept = total = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with in_path.open(encoding="utf-8") as src, out_path.open(
        "w", encoding="utf-8"
    ) as dst:
        for line in src:
            line = line.strip()
            if not line:
                continue
            total += 1
            record = json.loads(line)
            status, reason = classify(
                record, drop_cjk=drop_cjk, grounding=grounding
            )
            status_counts[status] += 1
            reason_counts[reason] += 1
            if status != "drop":
                dst.write(json.dumps(record, ensure_ascii=False) + "\n")
                kept += 1

    stats = {
        "total": total,
        "kept": kept,
        "dropped": total - kept,
        "by_status": dict(status_counts),
        "by_reason": dict(reason_counts),
    }
    print(
        f"[sist_clean] {in_path} -> {out_path}: kept {kept}/{total} "
        f"(dropped {total - kept})"
    )
    for status in ("keep", "cjk_keep", "flag", "drop"):
        if status_counts[status]:
            print(f"  status {status:9} {status_counts[status]}")
    for reason, count in reason_counts.most_common():
        print(f"  reason {reason:24} {count}")
    return stats


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Clean the sist silver QA pool.")
    parser.add_argument("--in", dest="in_path", default=str(DATA_DIR / "sist" / "qa_silver.jsonl"))
    parser.add_argument("--out", dest="out_path", default=str(DATA_DIR / "sist" / "qa_silver_clean.jsonl"))
    parser.add_argument("--corpus-dir", default="corpus/sist")
    parser.add_argument("--drop-cjk", action="store_true", help="Exclude CJK answers (pure-English EM).")
    parser.add_argument("--no-grounding", action="store_true", help="Skip the grounding re-check.")
    args = parser.parse_args(argv)

    clean_pool(
        args.in_path,
        args.out_path,
        drop_cjk=args.drop_cjk,
        verify_grounding=not args.no_grounding,
        corpus_dir=args.corpus_dir,
    )


if __name__ == "__main__":
    main()
