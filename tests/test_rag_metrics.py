"""Tests for RAG answer-quality metrics (no model weights required)."""

from rag.metrics import exact_match, normalize_answer, rouge_l, token_f1


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
