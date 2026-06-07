"""Tests for the test_results.csv deliverable export and its merge semantics."""

import csv

from rag.export_results import (
    accuracy_summary,
    build_test_rows,
    export_test_results,
)


def _fake_score(_q, answer, _ref):
    """Stub judge: raw absolute score 5 if the answer contains 'good', else 1."""
    return 5 if "good" in answer else 1


def _read(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def test_build_columns_for_enabled_baselines_only():
    rows = [
        {"qid": "q0", "question": "Q0", "method": "topk", "prediction": "bad", "reference": "R0"},
        {"qid": "q0", "question": "Q0", "method": "srag", "prediction": "good", "reference": "R0"},
    ]
    out, baselines = build_test_rows(rows, _fake_score)
    assert baselines == ["topk"]  # only the baseline actually present
    row = out[0]
    assert list(row) == [
        "query",
        "gt_answer",
        "sys_resp_before_opt_topk",
        "is_correct_before_opt_topk",
        "prometheus_score_before_opt_topk",
        "sys_resp_after_opt",
        "is_correct_after_opt",
        "prometheus_score_after_opt",
    ]
    assert row["query"] == "Q0" and row["gt_answer"] == "R0"
    # baseline "bad" -> raw score 1 -> below threshold 3 -> incorrect
    assert row["sys_resp_before_opt_topk"] == "bad"
    assert row["is_correct_before_opt_topk"] == 0
    assert row["prometheus_score_before_opt_topk"] == 1
    # S-RAG "good" -> raw score 5 -> at/above threshold -> correct
    assert row["sys_resp_after_opt"] == "good"
    assert row["is_correct_after_opt"] == 1
    assert row["prometheus_score_after_opt"] == 5
    # no greedy/mmr columns since those methods were not run
    assert not any("greedy" in c or "mmr" in c for c in row)


def test_empty_answer_is_incorrect_without_judging():
    rows = [{"qid": "q0", "question": "Q", "method": "topk", "prediction": "", "reference": "R"}]
    out, _ = build_test_rows(rows, _fake_score, baselines=["topk"])
    assert out[0]["is_correct_before_opt_topk"] == 0
    assert out[0]["prometheus_score_before_opt_topk"] == 0  # not judged -> raw 0


def test_merge_accumulates_methods_without_duplicate_file(tmp_path):
    run_dir = tmp_path
    # Run 1: baselines only (topk), no S-RAG yet.
    run1 = [
        {"qid": "q0", "question": "Q0", "method": "topk", "prediction": "bad", "reference": "R0"},
        {"qid": "q1", "question": "Q1", "method": "topk", "prediction": "good", "reference": "R1"},
    ]
    path1, _, _ = export_test_results(run1, run_dir, _fake_score, baselines=["topk"])

    # Run 2: same settings, now S-RAG (after optimization) + a new baseline mmr.
    run2 = [
        {"qid": "q0", "question": "Q0", "method": "srag", "prediction": "good", "reference": "R0"},
        {"qid": "q1", "question": "Q1", "method": "srag", "prediction": "bad", "reference": "R1"},
        {"qid": "q0", "question": "Q0", "method": "mmr", "prediction": "good", "reference": "R0"},
        {"qid": "q1", "question": "Q1", "method": "mmr", "prediction": "good", "reference": "R1"},
    ]
    path2, _, _ = export_test_results(run2, run_dir, _fake_score, baselines=["mmr"])

    assert path1 == path2  # same file, not a duplicate
    merged = _read(path1)
    assert len(merged) == 2  # still two questions, rows merged not appended

    by_q = {r["query"]: r for r in merged}
    q0 = by_q["Q0"]
    # topk column from run 1 is preserved (not clobbered by run 2)
    assert q0["sys_resp_before_opt_topk"] == "bad"
    assert q0["prometheus_score_before_opt_topk"] == "1"
    # mmr column added by run 2
    assert q0["sys_resp_before_opt_mmr"] == "good"
    assert q0["is_correct_before_opt_mmr"] == "1"
    assert q0["prometheus_score_before_opt_mmr"] == "5"
    # after-optimization (S-RAG) filled in by run 2
    assert q0["sys_resp_after_opt"] == "good"
    assert q0["is_correct_after_opt"] == "1"
    assert q0["prometheus_score_after_opt"] == "5"


def test_blank_after_does_not_clobber_existing(tmp_path):
    # Run 1 has S-RAG filled.
    run1 = [
        {"qid": "q0", "question": "Q0", "method": "srag", "prediction": "good", "reference": "R0"},
    ]
    path, _, _ = export_test_results(run1, tmp_path, _fake_score, baselines=[])
    # Run 2 is baseline-only (no S-RAG) => after_opt is blank for these rows.
    run2 = [
        {"qid": "q0", "question": "Q0", "method": "topk", "prediction": "bad", "reference": "R0"},
    ]
    export_test_results(run2, tmp_path, _fake_score, baselines=["topk"])
    q0 = _read(path)[0]
    assert q0["sys_resp_after_opt"] == "good"  # preserved, not blanked
    assert q0["prometheus_score_after_opt"] == "5"  # raw score preserved too
    assert q0["sys_resp_before_opt_topk"] == "bad"


def test_accuracy_summary_counts():
    rows = [
        {"qid": "q0", "question": "Q0", "method": "topk", "prediction": "good", "reference": "R0"},
        {"qid": "q0", "question": "Q0", "method": "srag", "prediction": "good", "reference": "R0"},
        {"qid": "q1", "question": "Q1", "method": "topk", "prediction": "bad", "reference": "R1"},
        {"qid": "q1", "question": "Q1", "method": "srag", "prediction": "good", "reference": "R1"},
    ]
    out, baselines = build_test_rows(rows, _fake_score)
    summary = {r["column"]: r for r in accuracy_summary(out, baselines)}
    assert summary["is_correct_before_opt_topk"]["correct"] == 1
    assert summary["is_correct_after_opt"]["correct"] == 2
