"""Export the per-question test-results CSV deliverable.

One row per test question. The "before optimization" responses are the enabled
baseline selection strategies (top-k / MMR / greedy), each as its own column;
the "after optimization" response is S-RAG. Binary correctness is judged by the
open Prometheus2 model (absolute grading vs the reference answer) — never a
commercial API.

Columns (baseline columns only appear for methods actually run):
  query, gt_answer,
  sys_resp_before_opt_<m>, is_correct_before_opt_<m>, prometheus_score_before_opt_<m>
                                                       for each baseline m
  sys_resp_after_opt, is_correct_after_opt, prometheus_score_after_opt   (S-RAG)

is_correct_* is 1 iff the Prometheus2 absolute score >= threshold (default 3);
prometheus_score_* is that raw 1-5 score (0 for an empty/failed answer).

Usage as a CLI to backfill an existing run from its results.csv:
  python3 -m rag.export_results ckpt/<model>/<run-name> \\
      --judge-model models/prometheus-7b-v2.0 [--threshold 3]
"""

from __future__ import annotations

import csv
from collections import OrderedDict
from collections.abc import Callable, Sequence
from pathlib import Path

BASELINE_ORDER = ("topk", "mmr", "greedy")  # canonical "before optimization" order
SRAG = "srag"  # "after optimization"

ScoreFn = Callable[[str, str, str], int]  # (question, answer, reference) -> raw score 1-5


def _pivot(rows: Sequence[dict]) -> "OrderedDict[str, dict]":
    """Group flat (qid, method) rows into one entry per question (order kept)."""
    by_qid: OrderedDict[str, dict] = OrderedDict()
    for r in rows:
        qid = r["qid"]
        entry = by_qid.setdefault(
            qid,
            {
                "question": r.get("question", ""),
                "reference": r.get("reference", ""),
                "pred": {},
            },
        )
        entry["pred"][r["method"]] = r.get("prediction", "")
    return by_qid


def _raw_score(score_fn: ScoreFn, question: str, answer: str, reference: str) -> int:
    """0 for an empty/failed answer; otherwise the judge's raw absolute score."""
    return score_fn(question, answer, reference) if answer and answer.strip() else 0


def build_test_rows(
    rows: Sequence[dict],
    score_fn: ScoreFn,
    baselines: Sequence[str] | None = None,
    threshold: int = 3,
) -> tuple[list[dict], list[str]]:
    """Pivot results into deliverable rows and return (rows, baseline columns).

    `score_fn` returns the judge's raw absolute score (1-5; 0 for an empty or
    failed answer). Each method contributes three columns: the response, a
    binary is_correct (raw score >= threshold), and that raw Prometheus2 score.
    """
    by_qid = _pivot(rows)
    present = {m for e in by_qid.values() for m in e["pred"]}
    if baselines is None:
        baselines = [m for m in BASELINE_ORDER if m in present]
    else:
        order = {m: i for i, m in enumerate(BASELINE_ORDER)}
        baselines = sorted(
            (m for m in baselines if m != SRAG and m in present),
            key=lambda m: order.get(m, len(order)),
        )

    # Only emit the S-RAG (after-opt) columns when S-RAG actually ran here, so a
    # baseline-only run does not overwrite an existing S-RAG result on merge (a
    # numeric 0 is not "blank" and would otherwise clobber a real score).
    has_srag = SRAG in present

    out: list[dict] = []
    for entry in by_qid.values():
        q, ref = entry["question"], entry["reference"]
        row: "OrderedDict[str, object]" = OrderedDict()
        row["query"] = q
        row["gt_answer"] = ref
        for m in baselines:  # before optimization: one triple of columns per baseline
            resp = entry["pred"].get(m, "")
            score = _raw_score(score_fn, q, resp, ref)
            row[f"sys_resp_before_opt_{m}"] = resp
            row[f"is_correct_before_opt_{m}"] = int(score >= threshold)
            row[f"prometheus_score_before_opt_{m}"] = score
        if has_srag:  # after optimization: S-RAG
            srag_resp = entry["pred"].get(SRAG, "")
            srag_score = _raw_score(score_fn, q, srag_resp, ref)
            row["sys_resp_after_opt"] = srag_resp
            row["is_correct_after_opt"] = int(srag_score >= threshold)
            row["prometheus_score_after_opt"] = srag_score
        out.append(dict(row))
    return out, list(baselines)


def _merge_rows(
    existing: Sequence[dict], new: Sequence[dict], merge_keys: Sequence[str]
) -> list[dict]:
    """Upsert new rows into existing by key, overlaying (and adding) columns.

    Same query => the row is updated: columns from this run win, columns from a
    prior run (e.g. a baseline only run earlier) are kept. New query => appended.
    """

    def key_of(row: dict) -> tuple:
        return tuple(str(row.get(k, "")) for k in merge_keys)

    merged: "OrderedDict[tuple, dict]" = OrderedDict(
        (key_of(r), dict(r)) for r in existing
    )
    for row in new:
        key = key_of(row)
        if key in merged:  # overlay, but never clobber a value with a blank
            for field, value in row.items():
                if value is None or value == "":
                    continue
                merged[key][field] = value
        else:
            merged[key] = dict(row)
    return list(merged.values())


def write_csv(
    rows: Sequence[dict],
    path: str | Path,
    merge_keys: Sequence[str] | None = ("query",),
) -> Path:
    path = Path(path)
    rows = list(rows)
    if merge_keys and path.exists() and path.stat().st_size > 0:
        with path.open(newline="") as handle:
            existing = list(csv.DictReader(handle))
        rows = _merge_rows(existing, rows, merge_keys)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, restval="")
        writer.writeheader()
        writer.writerows(rows)
    return path


def export_test_results(
    rows: Sequence[dict],
    run_dir: str | Path,
    score_fn: ScoreFn,
    baselines: Sequence[str] | None = None,
    threshold: int = 3,
    filename: str = "test_results.csv",
) -> tuple[Path, list[dict], list[str]]:
    """Build and write/merge test_results.csv; return (path, rows, baselines)."""
    test_rows, used = build_test_rows(rows, score_fn, baselines, threshold)
    path = write_csv(test_rows, Path(run_dir) / filename, merge_keys=("query",))
    return path, test_rows, used


def judge_raw_score_fn(judge) -> ScoreFn:
    """Wrap a loaded judge into a (q, a, ref) -> raw absolute score (1-5) function."""
    from .judge import absolute_score

    return lambda q, a, ref: absolute_score(judge, q, a, ref)


def accuracy_summary(test_rows: Sequence[dict], baselines: Sequence[str]) -> list[dict]:
    """Per-column correct/total over the deliverable rows (for a console table)."""
    cols = [f"is_correct_before_opt_{m}" for m in baselines] + ["is_correct_after_opt"]
    out: list[dict] = []
    for col in cols:
        vals = [int(r[col]) for r in test_rows if col in r and r[col] != ""]
        total = len(vals)
        correct = sum(vals)
        out.append(
            {
                "column": col,
                "correct": correct,
                "total": total,
                "accuracy": round(correct / total, 4) if total else 0.0,
            }
        )
    return out


def _read_results_csv(path: str | Path) -> list[dict]:
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def main(argv: Sequence[str] | None = None) -> None:
    import argparse

    from . import config

    parser = argparse.ArgumentParser(description="Backfill test_results.csv from a run.")
    parser.add_argument("run_dir", help="ckpt/<model>/<run-name> with results.csv")
    parser.add_argument("--judge-model", default=None)
    parser.add_argument("--threshold", type=int, default=3)
    args = parser.parse_args(argv)

    rows = _read_results_csv(Path(args.run_dir) / "results.csv")

    from .generator import LocalGenerator

    judge = LocalGenerator(args.judge_model or config.PAPER_MODELS["judge"])
    score_fn = judge_raw_score_fn(judge)
    path, test_rows, baselines = export_test_results(
        rows, args.run_dir, score_fn, threshold=args.threshold
    )

    print(f"wrote {path}  ({len(test_rows)} questions)")
    for r in accuracy_summary(test_rows, baselines):
        print(f"  {r['column']:30} acc {r['accuracy']:.4f}  ({r['correct']}/{r['total']})")


if __name__ == "__main__":
    main()
