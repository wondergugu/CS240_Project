"""SIST QA examples (silver questions + short reference answers).

sist has no human-labeled QA, so questions and gold answers are silver-generated
once by a local model reading sist passages (see rag/preprocess/sist_silver.py)
and stored as data/sist/qa_silver.jsonl. This loader only reads that pool; the
retrieval corpus is separate (corpus/sist, see rag/corpus.py) so candidate
passages never leak from the answer data.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from ..config import DATA_DIR
from .eli5 import QAExample

QA_FILENAME = "qa_silver.jsonl"
QA_CLEAN_FILENAME = "qa_silver_clean.jsonl"


def _resolve_pool_path(data_dir: str | Path | None) -> Path:
    """Prefer the cleaned pool (rag.preprocess.sist_clean) when it exists."""
    base = Path(data_dir or DATA_DIR) / "sist"
    clean = base / QA_CLEAN_FILENAME
    return clean if clean.is_file() else base / QA_FILENAME


def _references(record: dict) -> list[str]:
    refs = record.get("references")
    if isinstance(refs, list):
        refs = [r.strip() for r in refs if isinstance(r, str) and r.strip()]
        if refs:
            return refs
    answer = record.get("answer")
    return [answer.strip()] if isinstance(answer, str) and answer.strip() else []


def load_sist_qa(
    num_samples: int = 50,
    seed: int = 42,
    data_dir: str | Path | None = None,
    qa_path: str | Path | None = None,
) -> list[QAExample]:
    """Load up to `num_samples` sist (question, short reference answers) examples.

    Reads the silver QA pool and samples `num_samples` from it with `seed` so a
    run is reproducible and the expensive generation step is decoupled from the
    per-run sampling.
    """
    path = Path(qa_path) if qa_path else _resolve_pool_path(data_dir)
    if not path.is_file():
        raise FileNotFoundError(
            f"no sist silver QA at {path}; build it with rag.preprocess.sist_silver"
        )

    records: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    random.Random(seed).shuffle(records)
    examples: list[QAExample] = []
    for record in records:
        question = (record.get("question") or "").strip()
        references = _references(record)
        if not question or not references:
            continue
        examples.append(
            QAExample(
                qid=str(record.get("qid", record.get("source_pid", len(examples)))),
                question=question,
                references=tuple(references),
            )
        )
        if len(examples) >= num_samples:
            break
    return examples
