"""Tests for the per-run artifact recorder."""

import csv
import json

from rag.recorder import RunRecorder


def test_recorder_writes_config_and_results(tmp_path):
    rec = RunRecorder(tmp_path, "NousResearch/Meta-Llama-3-8B-Instruct", "eli5_srag_n5")
    rec.save_config({"dataset": "eli5", "method": "srag", "budget": 4096})
    rec.save_results(
        [
            {"qid": "q0", "method": "srag", "em": 1.0, "rougeL": 0.5},
            {"qid": "q1", "method": "srag", "em": 0.0, "rougeL": 0.2},
        ]
    )
    rec.save_best_params({"mmr_lambda": 0.5})

    run_dir = tmp_path / "nousresearch-meta-llama-3-8b-instruct" / "eli5_srag_n5"
    assert (run_dir / "config.json").exists()
    assert (run_dir / "config.yaml").exists()

    config = json.loads((run_dir / "config.json").read_text())
    assert config["dataset"] == "eli5"

    with (run_dir / "results.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert [r["qid"] for r in rows] == ["q0", "q1"]
    assert json.loads((run_dir / "best_params.json").read_text())["mmr_lambda"] == 0.5


def test_flat_layout_omits_model_parent(tmp_path):
    # group_by_model=False writes a single flat folder named by run_name (the
    # caller folds the model + settings hash into run_name) directly under ckpt/.
    rec = RunRecorder(
        tmp_path,
        "Qwen/Qwen2.5-0.5B-Instruct",
        "qwen-qwen2.5-0.5b-instruct_a1b2c3d4",
        group_by_model=False,
    )
    rec.save_config({"dataset": "sist"})
    assert rec.run_dir == tmp_path / "qwen-qwen2.5-0.5b-instruct_a1b2c3d4"
    assert (rec.run_dir / "config.json").exists()
    # no model-slug parent directory was created
    assert not (tmp_path / "qwen-qwen2.5-0.5b-instruct").exists()
