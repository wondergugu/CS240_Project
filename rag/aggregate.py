"""Aggregate ckpt run artifacts into Table 1 (metrics) and Table 2 (W/T/L)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .config import CKPT_DIR


def collect(ckpt_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_frames: list[pd.DataFrame] = []
    judge_frames: list[pd.DataFrame] = []
    for run_dir in sorted(Path(ckpt_dir).glob("*/*")):
        config_path = run_dir / "config.json"
        if not config_path.exists():
            continue
        config = json.loads(config_path.read_text())
        for filename, frames in (
            ("metrics_summary.csv", metric_frames),
            ("judge_wtl.csv", judge_frames),
        ):
            path = run_dir / filename
            if path.exists() and path.stat().st_size > 0:
                frame = pd.read_csv(path)
                frame.insert(0, "dataset", config.get("dataset"))
                frame.insert(1, "run", run_dir.name)
                frames.append(frame)

    metrics = pd.concat(metric_frames, ignore_index=True) if metric_frames else pd.DataFrame()
    judge = pd.concat(judge_frames, ignore_index=True) if judge_frames else pd.DataFrame()
    return metrics, judge


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt-dir", default=str(CKPT_DIR))
    args = parser.parse_args()

    metrics, judge = collect(args.ckpt_dir)
    print("=== Table 1: answer-quality metrics ===")
    print(metrics.to_string(index=False) if not metrics.empty else "(no metric runs found)")
    print("\n=== Table 2: S-RAG vs baselines (win/tie/loss) ===")
    print(judge.to_string(index=False) if not judge.empty else "(no judge runs found)")


if __name__ == "__main__":
    main()
