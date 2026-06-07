"""Per-run artifact logging into `ckpt/<model>/<run-id>/`.

Each run records its launch parameters (config.json + config.yaml), the
per-instance results (results.csv), aggregated metrics (metrics_summary.csv),
pairwise judge outcomes (judge_wtl.csv), and any tuned-hyperparameter "best"
artifact (best_params.json). Runs are grouped by generator model.

The paper performs no gradient training, so there are usually no trained
weights; `best_params.json` captures the closest notion of a "best" artifact
(e.g. an MMR lambda tuned on the dev split).
"""

from __future__ import annotations

import csv
import json
from collections import OrderedDict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml


def _slug(name: str) -> str:
    return name.strip().lower().replace("/", "-").replace(" ", "-")


class RunRecorder:
    def __init__(self, ckpt_dir: str | Path, gen_model: str, run_name: str):
        self.run_dir = Path(ckpt_dir) / _slug(gen_model) / run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def save_config(self, config: dict[str, Any]) -> Path:
        (self.run_dir / "config.json").write_text(
            json.dumps(config, indent=2, ensure_ascii=False, default=str)
        )
        (self.run_dir / "config.yaml").write_text(
            yaml.safe_dump(config, sort_keys=False, allow_unicode=True)
        )
        return self.run_dir / "config.json"

    @staticmethod
    def _read_csv(path: Path) -> list[dict[str, Any]]:
        with path.open(newline="") as handle:
            return list(csv.DictReader(handle))

    @staticmethod
    def _merge_rows(
        existing: Sequence[dict[str, Any]],
        new: Sequence[dict[str, Any]],
        merge_keys: Sequence[str],
    ) -> list[dict[str, Any]]:
        """Upsert new rows into existing by `merge_keys`, overlaying columns.

        A row whose key already exists is updated in place (new column values
        win, columns only in the old row are kept); a row with a new key is
        appended. This lets one run dir accumulate results across method subsets
        run separately (same model/data settings) instead of duplicating files.
        """

        def key_of(row: dict[str, Any]) -> tuple:
            return tuple(str(row.get(k, "")) for k in merge_keys)

        merged: "OrderedDict[tuple, dict[str, Any]]" = OrderedDict(
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

    def _write_csv(
        self,
        filename: str,
        rows: Sequence[dict[str, Any]],
        merge_keys: Sequence[str] | None = None,
    ) -> Path:
        path = self.run_dir / filename
        rows = list(rows)
        if merge_keys and path.exists() and path.stat().st_size > 0:
            rows = self._merge_rows(self._read_csv(path), rows, merge_keys)
        if not rows:
            path.write_text("")
            return path
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

    def save_results(self, rows: Sequence[dict[str, Any]]) -> Path:
        return self._write_csv("results.csv", rows, merge_keys=("qid", "method"))

    def save_summary(self, rows: Sequence[dict[str, Any]]) -> Path:
        return self._write_csv("metrics_summary.csv", rows, merge_keys=("method",))

    def save_judge(self, rows: Sequence[dict[str, Any]]) -> Path:
        return self._write_csv("judge_wtl.csv", rows, merge_keys=("comparison",))

    def save_best_params(self, params: dict[str, Any]) -> Path:
        path = self.run_dir / "best_params.json"
        path.write_text(json.dumps(params, indent=2, ensure_ascii=False, default=str))
        return path
