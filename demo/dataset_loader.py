"""Load paper QA datasets into document-selection examples.

The S-RAG implementation expects retrieved candidate documents with relevance
scores and token costs. The public QA datasets do not all ship with the same
retrieval format, so this module converts available dataset fields into a common
Query + Document candidate pool for reproducible selection experiments.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
import re
from typing import Any

import pyarrow.parquet as pq

from srag.types import Document, Query
from srag.utils import count_tokens

PAPER_SAMPLE_SIZE = 500
DEFAULT_DATA_ROOT = Path("data")
SUPPORTED_DATASETS = ("eli5", "hotpotqa")

_DATASET_FILE_PATTERNS = {
    "eli5": ("*.parquet", "**/*.parquet"),
    "hotpotqa": (
        "distractor/validation-*.parquet",
        "fullwiki/validation-*.parquet",
        "distractor/train-*.parquet",
        "fullwiki/train-*.parquet",
        "*.parquet",
        "**/*.parquet",
    ),
}

_QUERY_FIELDS = {
    "eli5": ("input", "question", "title", "query"),
    "hotpotqa": ("question", "query", "input"),
}

_ANSWER_FIELDS = ("answer", "answers", "output", "short_answers")
_SKIP_TEXT_KEYS = {
    "id",
    "qid",
    "query_id",
    "question",
    "question_text",
    "input",
    "title",
    "answer",
    "answers",
    "short_answers",
}
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_SPACE_RE = re.compile(r"\s+")


class DatasetNotFoundError(FileNotFoundError):
    """Raised when a requested paper dataset has no local Parquet files."""


@dataclass(frozen=True, slots=True)
class PaperDatasetExample:
    dataset: str
    query: Query
    candidates: list[Document]
    gold_doc_ids: frozenset[str]
    answer: str = ""


def available_datasets(data_root: str | Path = DEFAULT_DATA_ROOT) -> list[str]:
    root = Path(data_root)
    return [name for name in SUPPORTED_DATASETS if dataset_files(name, root)]


def dataset_files(dataset: str, data_root: str | Path = DEFAULT_DATA_ROOT) -> list[Path]:
    if dataset not in SUPPORTED_DATASETS:
        raise ValueError(f"unsupported dataset: {dataset!r}")

    root = Path(data_root) / dataset
    if not root.exists():
        return []

    files: list[Path] = []
    seen: set[Path] = set()
    for pattern in _DATASET_FILE_PATTERNS[dataset]:
        for path in sorted(root.glob(pattern)):
            if path.is_file() and path.suffix == ".parquet" and path not in seen:
                files.append(path)
                seen.add(path)
    return files


def missing_dataset_message(dataset: str, data_root: str | Path = DEFAULT_DATA_ROOT) -> str:
    return (
        f"No Parquet files found for {dataset!r} under "
        f"{Path(data_root) / dataset}. Run the dataset download commands "
        "recorded in README.md first."
    )


def load_paper_dataset(
    dataset: str,
    data_root: str | Path = DEFAULT_DATA_ROOT,
    sample_size: int = PAPER_SAMPLE_SIZE,
    seed: int = 42,
    max_candidates: int = 20,
    scan_multiplier: int = 20,
) -> list[PaperDatasetExample]:
    """Load up to `sample_size` examples from a paper dataset.

    The default `sample_size` is 500 to match the paper-level evaluation sample
    size requested for this project. Rows are sampled deterministically from the
    scanned prefix of the available local Parquet files.
    """
    if dataset not in SUPPORTED_DATASETS:
        raise ValueError(f"unsupported dataset: {dataset!r}")
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    if max_candidates <= 0:
        raise ValueError("max_candidates must be positive")

    files = dataset_files(dataset, data_root)
    if not files:
        raise DatasetNotFoundError(missing_dataset_message(dataset, data_root))

    rows = _read_scanned_rows(files, sample_size, scan_multiplier)
    rng = random.Random(_stable_seed(dataset, seed))
    rng.shuffle(rows)

    examples: list[PaperDatasetExample] = []
    for row_index, row in enumerate(rows):
        example = _row_to_example(dataset, row, row_index, max_candidates)
        if example is not None:
            examples.append(example)
        if len(examples) >= sample_size:
            break

    return examples


def _read_scanned_rows(
    files: list[Path], sample_size: int, scan_multiplier: int
) -> list[dict[str, Any]]:
    limit = max(sample_size, sample_size * max(1, scan_multiplier))
    rows: list[dict[str, Any]] = []

    for path in files:
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=512):
            rows.extend(batch.to_pylist())
            if len(rows) >= limit:
                return rows[:limit]
    return rows


def _row_to_example(
    dataset: str,
    row: dict[str, Any],
    row_index: int,
    max_candidates: int,
) -> PaperDatasetExample | None:
    query_text = _first_field_text(row, _QUERY_FIELDS[dataset])
    if not query_text:
        return None

    answer_text = _extract_answer(row)
    raw_candidates = _extract_candidate_texts(dataset, row, query_text, answer_text)
    if not raw_candidates:
        return None

    documents: list[Document] = []
    gold_doc_ids: set[str] = set()
    seen_texts: set[str] = set()
    qid = _safe_id(_first_field_text(row, ("id", "qid", "query_id"))) or str(row_index)

    for rank, (text, is_gold) in enumerate(raw_candidates):
        cleaned = _clean_text(text)
        if not cleaned or cleaned == query_text or cleaned in seen_texts:
            continue
        seen_texts.add(cleaned)

        doc_id = f"{dataset}:{qid}:d{len(documents)}"
        score = _relevance_score(query_text, cleaned, is_gold, rank)
        documents.append(
            Document(
                doc_id=doc_id,
                text=cleaned,
                relevance_score=score,
                token_count=max(1, count_tokens(cleaned)),
            )
        )
        if is_gold:
            gold_doc_ids.add(doc_id)
        if len(documents) >= max_candidates:
            break

    if not documents:
        return None

    return PaperDatasetExample(
        dataset=dataset,
        query=Query(text=query_text, query_id=f"{dataset}:{qid}"),
        candidates=documents,
        gold_doc_ids=frozenset(gold_doc_ids),
        answer=answer_text,
    )


def _extract_candidate_texts(
    dataset: str,
    row: dict[str, Any],
    query_text: str,
    answer_text: str,
) -> list[tuple[str, bool]]:
    if dataset == "hotpotqa":
        hotpot = _extract_hotpot_context(row)
        if hotpot:
            return hotpot

    texts: list[tuple[str, bool]] = []
    if dataset == "eli5":
        texts.extend(_extract_eli5_docs(row))

    if not texts:
        texts.extend(_extract_generic_texts(row, query_text, answer_text))

    return texts


def _extract_hotpot_context(row: dict[str, Any]) -> list[tuple[str, bool]]:
    context = row.get("context")
    supporting_titles = _supporting_titles(row.get("supporting_facts"))
    candidates: list[tuple[str, bool]] = []

    if isinstance(context, dict):
        titles = context.get("title") or context.get("titles") or []
        sentences = context.get("sentences") or context.get("sentence") or []
        for index, title in enumerate(titles):
            text_parts = _flatten_strings(sentences[index] if index < len(sentences) else [])
            text = " ".join(text_parts)
            if text:
                candidates.append((f"{title}. {text}" if title else text, title in supporting_titles))
    elif isinstance(context, list):
        for item in context:
            if isinstance(item, dict):
                title = str(item.get("title", ""))
                text = " ".join(_flatten_strings(item.get("sentences") or item.get("text") or item))
                if text:
                    candidates.append((f"{title}. {text}" if title else text, title in supporting_titles))
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                title = str(item[0])
                text = " ".join(_flatten_strings(item[1]))
                if text:
                    candidates.append((f"{title}. {text}" if title else text, title in supporting_titles))

    return candidates


def _extract_eli5_docs(row: dict[str, Any]) -> list[tuple[str, bool]]:
    candidates: list[tuple[str, bool]] = []
    output = row.get("output")

    for text in _flatten_output_answers(output):
        for chunk in _chunk_text(text, chunk_tokens=140, overlap=20):
            candidates.append((chunk, True))

    for text in _flatten_provenance_texts(output):
        for chunk in _chunk_text(text, chunk_tokens=140, overlap=20):
            candidates.append((chunk, True))

    return candidates


def _extract_generic_texts(
    row: dict[str, Any], query_text: str, answer_text: str
) -> list[tuple[str, bool]]:
    candidates: list[tuple[str, bool]] = []
    for key, value in row.items():
        if key in _SKIP_TEXT_KEYS:
            continue
        for text in _flatten_strings(value):
            if text and text != query_text:
                for chunk in _chunk_text(text, chunk_tokens=140, overlap=20):
                    candidates.append((chunk, _contains_answer(chunk, answer_text)))
    if answer_text:
        for chunk in _chunk_text(answer_text, chunk_tokens=140, overlap=20):
            candidates.append((chunk, True))
    return candidates


def _supporting_titles(value: Any) -> set[str]:
    titles: set[str] = set()
    if isinstance(value, dict):
        for title in value.get("title") or value.get("titles") or []:
            if title:
                titles.add(str(title))
    elif isinstance(value, list):
        for item in value:
            if isinstance(item, dict) and item.get("title"):
                titles.add(str(item["title"]))
            elif isinstance(item, (list, tuple)) and item:
                titles.add(str(item[0]))
    return titles


def _extract_answer(row: dict[str, Any]) -> str:
    for field in _ANSWER_FIELDS:
        if field not in row:
            continue
        texts = _flatten_output_answers(row[field]) or _flatten_strings(row[field])
        if texts:
            return _clean_text(texts[0])
    return ""


def _flatten_output_answers(value: Any) -> list[str]:
    answers: list[str] = []
    if isinstance(value, dict):
        for key in ("answer", "text", "value"):
            if isinstance(value.get(key), str):
                answers.append(value[key])
        for nested in ("answers", "output"):
            answers.extend(_flatten_output_answers(value.get(nested)))
    elif isinstance(value, list):
        for item in value:
            answers.extend(_flatten_output_answers(item))
    elif isinstance(value, str):
        answers.append(value)
    return [_clean_text(answer) for answer in answers if _clean_text(answer)]


def _flatten_provenance_texts(value: Any) -> list[str]:
    texts: list[str] = []
    if isinstance(value, dict):
        provenance = value.get("provenance")
        if provenance is not None:
            texts.extend(_flatten_strings(provenance))
        for nested in value.values():
            if isinstance(nested, (dict, list)):
                texts.extend(_flatten_provenance_texts(nested))
    elif isinstance(value, list):
        for item in value:
            texts.extend(_flatten_provenance_texts(item))
    return [_clean_text(text) for text in texts if _clean_text(text)]


def _first_field_text(row: dict[str, Any], fields: tuple[str, ...]) -> str:
    for field in fields:
        if field in row:
            texts = _flatten_strings(row[field])
            if texts:
                return _clean_text(texts[0])
    return ""


def _flatten_strings(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [_clean_text(value)] if _clean_text(value) else []
    if isinstance(value, bytes):
        text = value.decode("utf-8", errors="ignore")
        return [_clean_text(text)] if _clean_text(text) else []
    if isinstance(value, dict):
        texts: list[str] = []
        for nested in value.values():
            texts.extend(_flatten_strings(nested))
        return texts
    if isinstance(value, (list, tuple)):
        texts: list[str] = []
        for item in value:
            texts.extend(_flatten_strings(item))
        return texts
    return []


def _chunk_text(text: str, chunk_tokens: int, overlap: int) -> list[str]:
    cleaned = _clean_text(text)
    if not cleaned:
        return []
    words = cleaned.split()
    if len(words) <= chunk_tokens:
        return [cleaned]

    chunks: list[str] = []
    step = max(1, chunk_tokens - overlap)
    for start in range(0, len(words), step):
        chunk = " ".join(words[start : start + chunk_tokens])
        if chunk:
            chunks.append(chunk)
        if start + chunk_tokens >= len(words):
            break
    return chunks


def _relevance_score(query_text: str, doc_text: str, is_gold: bool, rank: int) -> float:
    query_terms = set(_terms(query_text))
    doc_terms = set(_terms(doc_text))
    overlap = len(query_terms & doc_terms) / max(1, len(query_terms))
    rank_prior = max(0.0, 0.2 - 0.005 * rank)
    gold_bonus = 1.0 if is_gold else 0.0
    return gold_bonus + overlap + rank_prior + 1e-6


def _contains_answer(text: str, answer: str) -> bool:
    if not answer:
        return False
    return answer.lower() in text.lower()


def _terms(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _clean_text(text: str) -> str:
    return _SPACE_RE.sub(" ", str(text)).strip()


def _safe_id(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.:-]+", "_", text)[:80]


def _stable_seed(dataset: str, seed: int) -> int:
    offset = sum((index + 1) * ord(char) for index, char in enumerate(dataset))
    return seed + offset
