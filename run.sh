#!/usr/bin/env bash
# Unified driver for end-to-end RAG evaluation (S-RAG vs MMR / Top-k / Greedy).
#
# Strategy, dataset, sample size, budget, etc. are controlled by the variables
# below (override via environment), which map onto run_rag.py's argparse flags.
# All models are local / open-weight; the system never calls a commercial LLM API.
set -euo pipefail
cd "$(dirname "$0")"

export HF_HOME="${HF_HOME:-$(pwd)/.hf_cache}"
export TOKENIZERS_PARALLELISM=false

# -------- configurable knobs (override via env) --------
DATASET="${DATASET:-eli5}"
METHODS="${METHODS:-topk mmr greedy srag}" # topk mmr greedy srag
NUM_SAMPLES="${NUM_SAMPLES:-50}"
BUDGET="${BUDGET:-4096}"
TOP_N="${TOP_N:-100}"
TOP_L="${TOP_L:-20}" # default 20
SEED="${SEED:-42}"
SMOKE="${SMOKE:-0}"              # 1 = small validation models, 0 = paper models
DEVICE="${DEVICE:-cuda:0}"      # 唯一的卡开关:cuda:0 / cuda:1 / cpu。改这一行即可,无需在终端动 CUDA_VISIBLE_DEVICES
# Pages to ingest when building a corpus. Default 1000; tune per dataset as the
# real corpus requires (e.g. CORPUS_PAGES=2000 DATASET=hotpotqa ./run.sh).
CORPUS_PAGES="${CORPUS_PAGES:-1000}"
CORPUS_DIR="${CORPUS_DIR:-corpus/${DATASET}}"

# sist-only: silver QA pool. sist has no labeled QA, so a pool is generated once
# by a local model (Prometheus2 by default; smoke uses the small gen model) and
# then reused/sampled by every run. See record.md for the rationale + caveats.
if [ "${SMOKE}" = "1" ]; then
  SIST_QA_MODEL="${SIST_QA_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
  SIST_QA_POOL="${SIST_QA_POOL:-20}"
else
  SIST_QA_MODEL="${SIST_QA_MODEL:-prometheus-eval/prometheus-7b-v2.0}"
  SIST_QA_POOL="${SIST_QA_POOL:-200}"
fi

# -------- 1. build the retrieval corpus if missing --------
if [ ! -s "${CORPUS_DIR}/passages.jsonl" ]; then
  if [ "${DATASET}" = "sist" ]; then
    echo "[run.sh] building ${CORPUS_DIR} from local data/sist chunks (no network)"
    python3 -c "from rag.corpus import build_sist_corpus; n,_=build_sist_corpus('${CORPUS_DIR}'); print('[run.sh] sist passages:', n)"
  else
    echo "[run.sh] building ${CORPUS_DIR} from KILT Wikipedia (${CORPUS_PAGES} pages)"
    python3 -c "from rag.corpus import build_kilt_corpus; build_kilt_corpus('${CORPUS_DIR}', max_pages=${CORPUS_PAGES})"
  fi
fi

# -------- 1.5 GPU 可见性自愈(让 DEVICE 成为唯一开关)--------
# 以后只改上面的 DEVICE=cuda:0 / cuda:1 即可,无需在终端 unset/设置任何变量。
# 原理:启动终端若残留了 CUDA_VISIBLE_DEVICES,会把 GPU 重新编号,导致 cuda:1 越界崩
# (invalid device ordinal)。这里统一按 DEVICE "钉定"物理卡,使 DEVICE=cuda:N 永远
# == nvidia-smi 里的物理第 N 张 GPU。
export CUDA_DEVICE_ORDER=PCI_BUS_ID   # 让 CUDA 的编号与 nvidia-smi 一致,映射可预期
RUN_DEVICE="${DEVICE}"
if [[ "${DEVICE}" == cuda:* ]]; then
  GPU_IDX="${DEVICE#cuda:}"
  NPHYS=0
  if command -v nvidia-smi >/dev/null 2>&1; then
    NPHYS=$(nvidia-smi -L 2>/dev/null | wc -l) || NPHYS=0
  fi
  NPHYS="${NPHYS:-0}"
  if ! [[ "${GPU_IDX}" =~ ^[0-9]+$ ]]; then
    echo "[run.sh] 错误:DEVICE='${DEVICE}' 格式不对,应为 cuda:0 / cuda:1 / cpu。" >&2; exit 1
  elif [ "${NPHYS}" -eq 0 ]; then
    echo "[run.sh] 警告:未检测到物理 GPU,程序将按需回退 CPU。" >&2
  elif [ "${GPU_IDX}" -ge "${NPHYS}" ]; then
    echo "[run.sh] 错误:DEVICE=${DEVICE} 越界 —— 本机仅 ${NPHYS} 张 GPU(可用 cuda:0..cuda:$((NPHYS-1)))。请把 DEVICE 改回范围内。" >&2; exit 1
  else
    export CUDA_VISIBLE_DEVICES="${GPU_IDX}"   # 只暴露这一张物理卡
    RUN_DEVICE="cuda:0"                         # 钉定后进程内只剩它,固定记作 cuda:0
    echo "[run.sh] GPU 钉定:DEVICE=${DEVICE} -> CUDA_VISIBLE_DEVICES=${GPU_IDX}(进程内 ${RUN_DEVICE} = 物理第 ${GPU_IDX} 张)"
  fi
elif [[ "${DEVICE}" == "cuda" ]]; then
  unset CUDA_VISIBLE_DEVICES 2>/dev/null || true   # 用全部可见卡,交给 torch 选默认
fi

# -------- 1.6 sist: build the silver QA pool if missing (needs the GPU) --------
# Runs after GPU pinning so the generator uses the same RUN_DEVICE. sist only.
if [ "${DATASET}" = "sist" ] && [ ! -s "data/sist/qa_silver.jsonl" ]; then
  [ "${SMOKE}" = "1" ] && export HF_HUB_OFFLINE=1   # smoke models are pre-downloaded
  echo "[run.sh] building data/sist/qa_silver.jsonl with ${SIST_QA_MODEL} (pool=${SIST_QA_POOL})"
  python3 -m rag.preprocess.sist_silver \
    --corpus-dir "${CORPUS_DIR}" \
    --gen-model "${SIST_QA_MODEL}" \
    --device "${RUN_DEVICE}" \
    --pool "${SIST_QA_POOL}" \
    --seed "${SEED}"
fi

# -------- 2. run the evaluation --------
SMOKE_FLAG=""
if [ "${SMOKE}" = "1" ]; then
  SMOKE_FLAG="--smoke"
  export HF_HUB_OFFLINE=1      # smoke models are pre-downloaded; stay offline
fi

echo "[run.sh] dataset=${DATASET} methods='${METHODS}' n=${NUM_SAMPLES} budget=${BUDGET} smoke=${SMOKE} device=${DEVICE} (run as ${RUN_DEVICE})"
python3 run_rag.py \
  --dataset "${DATASET}" \
  --methods ${METHODS} \
  --num-samples "${NUM_SAMPLES}" \
  --budget "${BUDGET}" \
  --top-n "${TOP_N}" \
  --top-l "${TOP_L}" \
  --seed "${SEED}" \
  --device "${RUN_DEVICE}" \
  --corpus-dir "${CORPUS_DIR}" \
  ${SMOKE_FLAG}
