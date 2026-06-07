"""ELI5 QA examples (questions + reference long-form answers) from KILT parquet.

Only the question and gold answers are taken from data/eli5; retrieval happens
against the independent KILT Wikipedia corpus (rag/corpus.py).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq

from ..config import DATA_DIR


@dataclass(frozen=True, slots=True)
class QAExample:
    qid: str
    question: str
    references: tuple[str, ...]


def _reference_answers(output) -> list[str]:
    references: list[str] = []
    if isinstance(output, list):
        for item in output:
            if isinstance(item, dict):
                answer = item.get("answer")
                if isinstance(answer, str) and answer.strip():
                    references.append(answer.strip())
    return references


def load_eli5_qa(
    num_samples: int = 50,
    seed: int = 42,
    split: str = "validation",
    data_dir: str | Path | None = None,
) -> list[QAExample]:
    """Load up to `num_samples` ELI5 (question, reference answers) examples."""
    root = Path(data_dir or DATA_DIR) / "eli5"
    files = sorted(root.glob(f"{split}-*.parquet")) or sorted(root.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no ELI5 parquet under {root}")

    scan_limit = max(num_samples * 20, num_samples)
    rows: list[dict] = []
    for file_path in files:
        for batch in pq.ParquetFile(file_path).iter_batches(batch_size=512):
            rows.extend(batch.to_pylist())
            if len(rows) >= scan_limit:
                break
        if len(rows) >= scan_limit:
            break

    random.Random(seed).shuffle(rows)
    examples: list[QAExample] = []
    for row in rows:
        question = (row.get("input") or "").strip()
        references = _reference_answers(row.get("output"))
        if not question or not references:
            continue
        examples.append(
            QAExample(
                qid=str(row.get("id", len(examples))),
                question=question,
                references=tuple(references),
            )
        )
        if len(examples) >= num_samples:
            break
    return examples
