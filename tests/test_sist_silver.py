"""Tests for sist silver-QA generation: parsing, grounding, and pool building."""

import json

from rag.corpus import build_sist_corpus  # noqa: F401 - kept for parity/imports
from rag.preprocess.sist_silver import build_silver_qa, parse_qa


def test_parse_qa_accepts_grounded_short_answer():
    src = "The dean of SIST studied at Berkeley before returning."
    assert parse_qa("Question: Where did the dean study?\nAnswer: Berkeley", src) == (
        "Where did the dean study?",
        "Berkeley",
    )


def test_parse_qa_rejects_ungrounded_overlong_or_malformed():
    src = "SIST is a school at ShanghaiTech University."
    # answer not present in the passage -> grounding failure
    assert parse_qa("Question: Where?\nAnswer: MIT", src) is None
    # missing question mark
    assert parse_qa("Question: Tell me\nAnswer: SIST", src) is None
    # answer too long (chars)
    assert parse_qa(f"Question: What?\nAnswer: {'x' * 80}", src) is None
    # no Answer line at all
    assert parse_qa("Question: What is SIST?", src) is None


class StubGenerator:
    """Echoes the passage's first word as a grounded short answer."""

    def chat(self, user, system=None, max_new_tokens=64):
        passage = user.split("Passage:\n", 1)[1].split("\n\n", 1)[0]
        first = passage.split()[0]
        return f"Question: What about {first}?\nAnswer: {first}"


def _write_passages(corpus_dir, texts):
    corpus_dir.mkdir(parents=True)
    with (corpus_dir / "passages.jsonl").open("w", encoding="utf-8") as handle:
        for i, text in enumerate(texts):
            handle.write(
                json.dumps({"pid": f"sist:{i}", "title": "", "text": text}) + "\n"
            )


def test_build_pool_dedups_and_grounds(tmp_path):
    corpus = tmp_path / "corpus"
    _write_passages(
        corpus,
        [
            "Alpha leads the systems security center at the school.",
            "Bravo manages the robotics laboratory and its students.",
            "Alpha also advises the undergraduate admissions committee.",  # dup question
        ],
    )
    out = tmp_path / "qa_silver.jsonl"
    count, path = build_silver_qa(
        corpus, out_path=out, generator=StubGenerator(), pool=10, min_chars=10
    )
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert count == len(records) == 2  # "What about Alpha?" deduped to one
    questions = {r["question"] for r in records}
    assert questions == {"What about Alpha?", "What about Bravo?"}
    for r in records:
        assert r["references"] == [r["answer"]]
        assert r["answer"] in {"Alpha", "Bravo"}


def test_build_pool_respects_pool_cap(tmp_path):
    corpus = tmp_path / "corpus"
    _write_passages(
        corpus,
        ["Alpha one two three four five.", "Bravo one two three four five."],
    )
    out = tmp_path / "qa.jsonl"
    count, _ = build_silver_qa(
        corpus, out_path=out, generator=StubGenerator(), pool=1, min_chars=10
    )
    assert count == 1  # stops at the pool cap
