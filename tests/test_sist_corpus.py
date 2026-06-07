"""Tests for the local sist corpus builder (data/sist chunks -> passages.jsonl)."""

import json

from rag.corpus import build_sist_corpus, load_passages


def _write_chunks(data_dir, chunks):
    jsonl_dir = data_dir / "jsonl"
    jsonl_dir.mkdir(parents=True)
    with (jsonl_dir / "chunks.jsonl").open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk, ensure_ascii=False) + "\n")


def test_schema_and_min_chars_filter(tmp_path):
    data_dir, out_dir = tmp_path / "data", tmp_path / "corpus"
    long_text = "ShanghaiTech SIST offers strong programs in computer science. " * 3
    _write_chunks(
        data_dir,
        [
            {"id": "1", "title": "T1", "url": "u1", "category": "faculty", "text": long_text},
            {"id": "2", "title": "T2", "url": "u2", "category": "news", "text": "too short"},
        ],
    )

    written, path = build_sist_corpus(out_dir, data_dir=data_dir, min_chars=80)
    assert written == 1  # the short chunk is dropped
    assert path == out_dir / "passages.jsonl"

    passages = load_passages(out_dir)
    assert len(passages) == 1
    record = passages[0]
    assert set(record) == {"pid", "title", "text", "url", "category"}
    assert record["pid"] == "sist:1"
    assert record["title"] == "T1" and record["category"] == "faculty"


def test_nav_lines_are_trimmed(tmp_path):
    data_dir, out_dir = tmp_path / "data", tmp_path / "corpus"
    body = "Real content about the dean and the research center and its labs. " * 2
    _write_chunks(
        data_dir,
        [{"id": "9", "title": "Home", "text": f"English\n导航\n学校首页\n{body}"}],
    )

    build_sist_corpus(out_dir, data_dir=data_dir, min_chars=40)
    text = load_passages(out_dir)[0]["text"]
    assert "导航" not in text and "学校首页" not in text
    assert "Real content about the dean" in text
