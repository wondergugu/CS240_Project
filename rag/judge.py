"""Open-model pairwise LLM judge with order-swap de-biasing (W/T/L).

Compares S-RAG answers against each baseline on the same question, judging
correctness / completeness / conciseness. Each pair is judged twice with the
answer order swapped; if the two verdicts disagree it is counted as a tie
(paper §4.1 / C.2). The judge is an open instruct model (Prometheus2 for the
paper, the smoke generator for validation) — never a commercial API.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Sequence

# Prometheus2 native relative-grading prompt (prometheus-eval format).
_RUBRIC = (
    "Which response more correctly, completely, and concisely answers the "
    "question? A better response is factually correct, covers the key points, "
    "and avoids irrelevant or redundant content."
)
_PROMPT = (
    "###Task Description:\n"
    "An instruction, two responses (Response A and Response B), and a score "
    "rubric are given.\n"
    "1. Write a brief feedback assessing the two responses strictly per the rubric.\n"
    "2. Then choose the better response, A or B.\n"
    '3. Output format: "Feedback: (feedback) [RESULT] (A or B)".\n'
    "4. Do not write anything after the result letter.\n\n"
    "###Instruction:\n{question}\n\n"
    "###Response A:\n{answer_a}\n\n"
    "###Response B:\n{answer_b}\n\n"
    "###Score Rubric:\n{rubric}\n\n"
    "###Feedback:\n"
)
_RESULT_RE = re.compile(r"\[RESULT\]\s*\(?\s*([AB])", re.IGNORECASE)


def _parse_verdict(text: str) -> str:
    match = _RESULT_RE.search(text)
    if match:
        return match.group(1).upper()
    tail = text.strip()[-4:].upper()  # fallback: a lone trailing A/B
    if "A" in tail and "B" not in tail:
        return "A"
    if "B" in tail and "A" not in tail:
        return "B"
    return "tie"


def judge_once(judge, question: str, answer_a: str, answer_b: str) -> str:
    prompt = _PROMPT.format(
        question=question,
        answer_a=answer_a or "(empty)",
        answer_b=answer_b or "(empty)",
        rubric=_RUBRIC,
    )
    return _parse_verdict(judge.chat(prompt, max_new_tokens=256))


def pairwise_outcome(judge, question: str, srag_answer: str, baseline_answer: str) -> str:
    """Win/tie/loss of S-RAG vs a baseline, with order-swap de-biasing."""
    # order 1: A = S-RAG, B = baseline; order 2: A = baseline, B = S-RAG.
    r1 = judge_once(judge, question, srag_answer, baseline_answer)
    r2 = judge_once(judge, question, baseline_answer, srag_answer)
    srag_preferred = (r1 == "A") + (r2 == "B")
    baseline_preferred = (r1 == "B") + (r2 == "A")
    if srag_preferred == 2:
        return "win"
    if baseline_preferred == 2:
        return "loss"
    return "tie"  # disagreement across orders or any tie => tie


def judge_run(
    judge,
    rows: Sequence[dict],
    baselines: Sequence[str] = ("topk", "mmr", "greedy"),
    srag_method: str = "srag",
) -> list[dict]:
    """Compute S-RAG vs each baseline W/T/L over the per-instance rows."""
    predictions: dict[str, dict[str, str]] = defaultdict(dict)
    questions: dict[str, str] = {}
    for row in rows:
        predictions[row["qid"]][row["method"]] = row.get("prediction", "")
        questions[row["qid"]] = row.get("question", "")

    summary: list[dict] = []
    for baseline in baselines:
        win = tie = loss = 0
        for qid, by_method in predictions.items():
            if srag_method not in by_method or baseline not in by_method:
                continue
            outcome = pairwise_outcome(
                judge, questions[qid], by_method[srag_method], by_method[baseline]
            )
            win += outcome == "win"
            tie += outcome == "tie"
            loss += outcome == "loss"
        total = win + tie + loss
        summary.append(
            {
                "comparison": f"srag_vs_{baseline}",
                "win": win,
                "tie": tie,
                "loss": loss,
                "win_pct": round(100 * win / total, 1) if total else 0.0,
                "tie_pct": round(100 * tie / total, 1) if total else 0.0,
                "loss_pct": round(100 * loss / total, 1) if total else 0.0,
            }
        )
    return summary


# --- Absolute (single-response) grading: per-answer correctness ---------------
# Prometheus2 native absolute-grading prompt (prometheus-eval format). Used to
# label each system response 1 (correct) / 0 (incorrect) against the reference,
# for the test_results.csv deliverable.
_ABS_RUBRIC = (
    "Does the response correctly and completely answer the question, "
    "consistent with the reference answer?\n"
    "Score 1: Largely incorrect or irrelevant; fails to answer the question.\n"
    "Score 2: Partially correct but with major factual errors or omissions.\n"
    "Score 3: Mostly correct and on-topic, with only minor errors or gaps.\n"
    "Score 4: Correct and complete, well aligned with the reference answer.\n"
    "Score 5: Fully correct, complete, and consistent with the reference answer."
)
_ABS_PROMPT = (
    "###Task Description:\n"
    "An instruction, a response to evaluate, a reference answer that gets a "
    "score of 5, and a score rubric are given.\n"
    "1. Write a brief feedback assessing the response strictly per the rubric.\n"
    "2. Then give an integer score from 1 to 5 based on the rubric.\n"
    '3. Output format: "Feedback: (feedback) [RESULT] (an integer 1-5)".\n'
    "4. Do not write anything after the score.\n\n"
    "###The instruction to evaluate:\n{question}\n\n"
    "###Response to evaluate:\n{answer}\n\n"
    "###Reference Answer (Score 5):\n{reference}\n\n"
    "###Score Rubric:\n{rubric}\n\n"
    "###Feedback:\n"
)
_SCORE_RE = re.compile(r"\[RESULT\]\s*\(?\s*([1-5])")


def _parse_score(text: str) -> int:
    match = _SCORE_RE.search(text)
    if match:
        return int(match.group(1))
    digits = re.findall(r"[1-5]", text)  # fallback: last standalone 1-5 digit
    return int(digits[-1]) if digits else 0


def absolute_score(judge, question: str, answer: str, reference: str) -> int:
    """Prometheus2 absolute grade (1-5) of one response against a reference."""
    prompt = _ABS_PROMPT.format(
        question=question,
        answer=answer or "(empty)",
        reference=reference or "(empty)",
        rubric=_ABS_RUBRIC,
    )
    return _parse_score(judge.chat(prompt, max_new_tokens=256))


def is_correct(
    judge, question: str, answer: str, reference: str, threshold: int = 3
) -> int:
    """1 if the response scores >= threshold under absolute grading, else 0."""
    return int(absolute_score(judge, question, answer, reference) >= threshold)
