#!/usr/bin/env bash
# Unified driver for end-to-end RAG evaluation (S-RAG vs MMR / Top-k / Greedy).
#
# SWEEP MODEL: every knob below EXCEPT METHODS is a space-separated LIST, and the
# run is the full Cartesian product
#     DATASET × NUM_SAMPLES × BUDGET × TOP_N × TOP_L × MMR_LAMBDA × SEED
# (e.g. TOP_N="10 20 50" ×3, TOP_L="10 20" ×2 → ×6). METHODS is NOT swept: the
# whole set is passed to each run and, sharing a settings hash, merges into the
# SAME folder. Each combination writes its own ckpt/<generator-slug>_<hash>/.
#
# This script only does DATA PREP (build each corpus once; build the sist silver
# QA pool once) and then hands the whole sweep to run_rag.py, which loads the
# heavy models ONCE and iterates the combinations (so big models are not reloaded
# per combination). All models are local / open-weight; never a commercial API.
set -euo pipefail
cd "$(dirname "$0")"

export HF_HOME="${HF_HOME:-$(pwd)/.hf_cache}"
export TOKENIZERS_PARALLELISM=false

# -------- swept knobs (space-separated lists; crossed by run_rag.py) --------
DATASET="${DATASET:-eli5 sist}"
NUM_SAMPLES="${NUM_SAMPLES:-100}"
BUDGET="${BUDGET:-4096}"
TOP_N="${TOP_N:-100}"        # 200/100
TOP_L="${TOP_L:-20}"
MMR_LAMBDA="${MMR_LAMBDA:-0.7}"
SEED="${SEED:-42}"

# -------- non-swept knobs --------
METHODS="${METHODS:-topk mmr greedy srag}" # passed whole to each run (NOT swept)
SMOKE="${SMOKE:-0}"            # 1 = small validation models, 0 = paper models
DEVICE="${DEVICE:-cuda:0}"     # 唯一的卡开关:cuda:0 / cuda:1 / cpu。改这一行即可
CORPUS_PAGES="${CORPUS_PAGES:-1000}"   # KILT pages when building a non-sist corpus

# sist-only: silver QA pool (generated once per dataset; sampled per run).
if [ "${SMOKE}" = "1" ]; then
  SIST_QA_MODEL="${SIST_QA_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
  SIST_QA_POOL="${SIST_QA_POOL:-20}"
else
  SIST_QA_MODEL="${SIST_QA_MODEL:-prometheus-eval/prometheus-7b-v2.0}"
  SIST_QA_POOL="${SIST_QA_POOL:-200}"
fi
SIST_QA_SEED="${SIST_QA_SEED:-42}"     # fixed seed for the one-time pool build

# -------- GPU 可见性自愈(DEVICE 是唯一开关;只做一次)--------
# 原理见 record.md:启动终端残留的 CUDA_VISIBLE_DEVICES 会重新编号 GPU,导致 cuda:1
# 越界崩(invalid device ordinal)。这里按 DEVICE 钉定物理卡,使 DEVICE=cuda:N 永远
# == nvidia-smi 的第 N 张,并把进程内设备固定记作 cuda:0(RUN_DEVICE)。
export CUDA_DEVICE_ORDER=PCI_BUS_ID
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

SMOKE_FLAG=""
if [ "${SMOKE}" = "1" ]; then
  SMOKE_FLAG="--smoke"
  export HF_HUB_OFFLINE=1      # smoke models are pre-downloaded; stay offline
fi

# sist silver pool must be at least as large as the biggest NUM_SAMPLES.
maxof() { local m=0 x; for x in $1; do x="${x%.*}"; if [ "$x" -gt "$m" ] 2>/dev/null; then m="$x"; fi; done; echo "$m"; }
NS_MAX=$(maxof "${NUM_SAMPLES}")
if [ "${NS_MAX}" -gt "${SIST_QA_POOL}" ]; then
  echo "[run.sh] bumping SIST_QA_POOL ${SIST_QA_POOL} -> ${NS_MAX} (= max NUM_SAMPLES)"
  SIST_QA_POOL="${NS_MAX}"
fi

# -------- DATA PREP: per dataset, build corpus once + sist silver QA once --------
for ds in ${DATASET}; do
  CORPUS_DIR="corpus/${ds}"
  if [ ! -s "${CORPUS_DIR}/passages.jsonl" ]; then
    if [ "${ds}" = "sist" ]; then
      echo "[run.sh] building ${CORPUS_DIR} from local data/sist chunks (no network)"
      python3 -c "from rag.corpus import build_sist_corpus; n,_=build_sist_corpus('${CORPUS_DIR}'); print('[run.sh] sist passages:', n)"
    else
      echo "[run.sh] building ${CORPUS_DIR} from KILT Wikipedia (${CORPUS_PAGES} pages)"
      python3 -c "from rag.corpus import build_kilt_corpus; build_kilt_corpus('${CORPUS_DIR}', max_pages=${CORPUS_PAGES})"
    fi
  fi
  if [ "${ds}" = "sist" ] && [ ! -s "data/sist/qa_silver.jsonl" ]; then
    echo "[run.sh] building data/sist/qa_silver.jsonl with ${SIST_QA_MODEL} (pool=${SIST_QA_POOL})"
    python3 -m rag.preprocess.sist_silver \
      --corpus-dir "${CORPUS_DIR}" \
      --gen-model "${SIST_QA_MODEL}" \
      --device "${RUN_DEVICE}" \
      --pool "${SIST_QA_POOL}" \
      --seed "${SIST_QA_SEED}" \
      ${SIST_QA_MAX_ATTEMPTS:+--max-attempts "${SIST_QA_MAX_ATTEMPTS}"}
  fi
  # Clean the silver pool once (drops PDF-glyph/formula garbage, re-verifies
  # grounding, keeps CJK). load_sist_qa prefers qa_silver_clean.jsonl when present.
  if [ "${ds}" = "sist" ] && [ -s "data/sist/qa_silver.jsonl" ] && [ ! -s "data/sist/qa_silver_clean.jsonl" ]; then
    echo "[run.sh] cleaning sist silver pool -> data/sist/qa_silver_clean.jsonl"
    python3 -m rag.preprocess.sist_clean \
      --in "data/sist/qa_silver.jsonl" \
      --out "data/sist/qa_silver_clean.jsonl" \
      --corpus-dir "${CORPUS_DIR}" \
      ${SIST_DROP_CJK:+--drop-cjk}
  fi
done

# -------- EVALUATION: hand the whole sweep to run_rag (models loaded once) --------
echo "[run.sh] launching sweep: datasets='${DATASET}' methods='${METHODS}' (device ${DEVICE}->${RUN_DEVICE})"
python3 run_rag.py \
  --dataset ${DATASET} \
  --methods ${METHODS} \
  --num-samples ${NUM_SAMPLES} \
  --budget ${BUDGET} \
  --top-n ${TOP_N} \
  --top-l ${TOP_L} \
  --mmr-lambda ${MMR_LAMBDA} \
  --seed ${SEED} \
  --device "${RUN_DEVICE}" \
  ${SMOKE_FLAG}
