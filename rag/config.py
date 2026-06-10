"""Central configuration for the end-to-end RAG evaluation layer.

All models are open-weight and run locally; the project never calls a
commercial LLM API (see the compliance note in README/record).
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
HF_CACHE = PROJECT_ROOT / ".hf_cache"
CORPUS_DIR = PROJECT_ROOT / "corpus"
CKPT_DIR = PROJECT_ROOT / "ckpt"
DATA_DIR = PROJECT_ROOT / "data"

# Route all Hugging Face downloads/loads to the project-local cache so weights
# live under the project (gitignored) and are reused across runs.
os.environ.setdefault("HF_HOME", str(HF_CACHE))

# --- Paper-faithful open models (self-hosted; no commercial APIs) ---
PAPER_MODELS = {
    "retriever": "BAAI/bge-large-en-v1.5",
    # Ungated mirror of meta-llama/Meta-Llama-3-8B-Instruct (identical weights).
    "generator": "NousResearch/Meta-Llama-3-8B-Instruct",
    "judge": "prometheus-eval/prometheus-7b-v2.0",
    "bertscore": "roberta-large",
}

# --- Small open models for fast end-to-end pipeline validation ("先跑通") ---
# Swapped in via --smoke / --gen-model until the paper weights finish downloading.
SMOKE_MODELS = {
    "retriever": "BAAI/bge-small-en-v1.5",
    "generator": "Qwen/Qwen2.5-0.5B-Instruct",
    "judge": "Qwen/Qwen2.5-0.5B-Instruct",
    "bertscore": "distilbert/distilbert-base-uncased",
}

# BERTScore backbones not in bert-score's built-in layer table need num_layers.
BERTSCORE_LAYERS = {
    "distilbert/distilbert-base-uncased": 6,
    "models/roberta-large": 17,  # paper backbone (local path); bert-score default layer
    "roberta-large": 17,
}

# Per-dataset generation length (paper Appendix B.1). sist is short-answer
# extractive (see generator._SHORT_INSTRUCTIONS), so it is capped tight so the
# model emits the answer span rather than a full paragraph that breaks EM.
MAX_NEW_TOKENS = {"eli5": 512, "hotpotqa": 192, "sist": 32}

# Short-answer datasets use EM/F1; long-form datasets use ROUGE-L/BERTScore.
SHORT_ANSWER_DATASETS = ("hotpotqa", "sist")
LONG_FORM_DATASETS = ("eli5",)

# sist has no human-labeled QA: a silver pool is generated once by a local model
# reading the sist corpus (rag/preprocess/sist_silver.py), then sampled per run.
SIST_QA_POOL = 200
SIST_QA_PATH = DATA_DIR / "sist" / "qa_silver.jsonl"
# Cleaned pool (rag/preprocess/sist_clean.py): drops PDF-glyph/formula garbage and
# re-verifies grounding, keeping CJK answers. Loaders prefer it when present.
SIST_QA_CLEAN_PATH = DATA_DIR / "sist" / "qa_silver_clean.jsonl"

# Selection / retrieval defaults (paper §4.1).
DEFAULT_BUDGET = 4096  # context budget measured in generator tokens
TOP_N = 200            # retrieved candidate-pool size per query
TOP_L = 20             # concept-universe depth for S-RAG / MMR proxy
MMR_LAMBDA = 0.7

DATASETS = ("eli5", "hotpotqa", "sist")
METHODS = ("srag", "mmr", "topk", "greedy")


def model_set(smoke: bool) -> dict[str, str]:
    """Return the active model registry (small smoke set vs paper set)."""
    return dict(SMOKE_MODELS if smoke else PAPER_MODELS)
