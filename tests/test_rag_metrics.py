"""Tests for RAG answer-quality metrics (no model weights required)."""

from rag.metrics import (
    exact_match,
    exact_match_contains,
    normalize_answer,
    rouge_l,
    token_f1,
)


def test_normalize_strips_articles_punctuation_case():
    assert normalize_answer("The Cat, sat!") == "cat sat"


def test_exact_match_articles_and_multi_gold():
    assert exact_match("The cat", "a cat") == 1.0
    assert exact_match("dog", ["cat", "dog"]) == 1.0
    assert exact_match("bird", ["cat", "dog"]) == 0.0
    assert exact_match("anything", []) == 0.0


def test_token_f1_partial_and_exact():
    assert token_f1("the quick brown fox", "quick brown fox") == 1.0
    assert 0.0 < token_f1("quick brown", "quick brown fox") < 1.0
    assert token_f1("", "fox") == 0.0


def test_rouge_l_identical_and_disjoint():
    assert rouge_l("the cat sat on the mat", "the cat sat on the mat") == 1.0
    assert 0.0 <= rouge_l("completely unrelated tokens", "the cat sat") < 1.0
    assert rouge_l("anything", []) == 0.0


def test_exact_match_contains_credits_answer_in_sentence():
    # Whole-string EM fails on a wrapped answer; containment EM credits it.
    pred = "According to the passages, the lab is the m&m lab."
    assert exact_match("m&m lab", pred) == 0.0  # whole-string (args: gold, pred)
    assert exact_match_contains(pred, "m&m lab") == 1.0
    assert exact_match_contains(pred, ["wrong lab", "m&m lab"]) == 1.0
    assert exact_match_contains("a totally different answer", "m&m lab") == 0.0
    # An empty gold must not vacuously match (empty string is a substring of all).
    assert exact_match_contains("anything at all", "") == 0.0
    assert exact_match_contains("anything", []) == 0.0


def test_cjk_exact_match_contains_and_f1():
    # CJK answers carry no spaces; char-level tokenization makes EM/F1 fire.
    pred = "根据文章，这门课程是医学影像物理基础。"
    assert exact_match_contains(pred, "医学影像物理基础") == 1.0
    # token_f1 now scores partial CJK overlap instead of collapsing to 0/1.
    score = token_f1("医学影像物理", "医学影像物理基础")
    assert 0.0 < score < 1.0
