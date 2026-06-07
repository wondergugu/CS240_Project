"""Answer-quality metrics for the RAG evaluation.

Short-answer QA (HotpotQA / sist) -> Exact Match (+ token F1).
Long-form QA (ELI5) -> ROUGE-L and BERTScore.

All metrics are computed locally; no external service is called. BERTScore
loads a local open scoring model on demand.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
import re
import string

from rouge_score import rouge_scorer

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_WHITESPACE = re.compile(r"\s+")
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def normalize_answer(text: str) -> str:
    """SQuAD-style normalization: lowercase, strip punctuation/articles/space."""
    lowered = text.lower().translate(_PUNCT_TABLE)
    no_articles = _ARTICLES.sub(" ", lowered)
    return _WHITESPACE.sub(" ", no_articles).strip()


def _gold_list(golds: str | Sequence[str]) -> list[str]:
    if isinstance(golds, str):
        return [golds] if golds else []
    return [g for g in golds if g]


def exact_match(prediction: str, golds: str | Sequence[str]) -> float:
    """1.0 if the normalized prediction equals any normalized gold, else 0.0."""
    references = _gold_list(golds)
    if not references:
        return 0.0
    pred = normalize_answer(prediction)
    return max(1.0 if pred == normalize_answer(g) else 0.0 for g in references)


def _f1(pred_tokens: list[str], gold_tokens: list[str]) -> float:
    if not pred_tokens or not gold_tokens:
        return 1.0 if pred_tokens == gold_tokens else 0.0
    overlap = sum((Counter(pred_tokens) & Counter(gold_tokens)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def token_f1(prediction: str, golds: str | Sequence[str]) -> float:
    """Best token-level F1 of the prediction against any gold answer."""
    references = _gold_list(golds)
    if not references:
        return 0.0
    pred_tokens = normalize_answer(prediction).split()
    return max(_f1(pred_tokens, normalize_answer(g).split()) for g in references)


_ROUGE_SCORER: rouge_scorer.RougeScorer | None = None


def rouge_l(prediction: str, golds: str | Sequence[str]) -> float:
    """Best ROUGE-L F-measure of the prediction against any gold answer."""
    global _ROUGE_SCORER
    if _ROUGE_SCORER is None:
        _ROUGE_SCORER = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    references = _gold_list(golds)
    if not references:
        return 0.0
    return max(
        _ROUGE_SCORER.score(g, prediction)["rougeL"].fmeasure for g in references
    )


def bertscore_f1(
    predictions: Sequence[str],
    references: Sequence[str],
    model_type: str = "roberta-large",
    lang: str = "en",
    batch_size: int = 32,
    num_layers: int | None = None,
) -> list[float]:
    """BERTScore F1 per (prediction, single-reference) pair via a local model."""
    from bert_score import score as _bert_score

    _, _, f1 = _bert_score(
        list(predictions),
        list(references),
        model_type=model_type,
        num_layers=num_layers,
        lang=lang,
        batch_size=batch_size,
        rescale_with_baseline=False,
        verbose=False,
    )
    return [float(x) for x in f1]
