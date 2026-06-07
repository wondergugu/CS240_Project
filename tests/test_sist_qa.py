"""Tests for the sist silver-QA loader (qa_silver.jsonl -> QAExample)."""

import json

from rag.preprocess.sist import load_sist_qa


def _write_qa(path, records):
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def test_loads_examples_and_reference_fallback(tmp_path):
    qa = tmp_path / "qa_silver.jsonl"
    _write_qa(
        qa,
        [
            {"qid": "x0", "question": "Who is the dean?", "references": ["A", "B"]},
            {"qid": "x1", "question": "Where is SIST?", "answer": "ShanghaiTech"},
            {"qid": "x2", "question": "", "answer": "skipme"},  # no question -> skipped
        ],
    )
    examples = load_sist_qa(num_samples=10, seed=1, qa_path=qa)
    by_q = {e.question: e for e in examples}
    assert len(examples) == 2  # the empty-question record is dropped
    assert by_q["Who is the dean?"].references == ("A", "B")
    assert by_q["Where is SIST?"].references == ("ShanghaiTech",)  # fallback to answer


def test_sampling_is_seed_deterministic(tmp_path):
    qa = tmp_path / "qa_silver.jsonl"
    _write_qa(
        qa,
        [{"qid": f"x{i}", "question": f"Q{i}?", "answer": f"A{i}"} for i in range(20)],
    )
    a = [e.qid for e in load_sist_qa(num_samples=5, seed=7, qa_path=qa)]
    b = [e.qid for e in load_sist_qa(num_samples=5, seed=7, qa_path=qa)]
    c = [e.qid for e in load_sist_qa(num_samples=5, seed=8, qa_path=qa)]
    assert a == b and len(a) == 5  # same seed -> same sample
    assert a != c  # a different seed samples differently
