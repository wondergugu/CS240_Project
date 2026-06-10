"""Streamlit demo: ask the SIST knowledge base with S-RAG.

    streamlit run app.py

The heavy models load once (cached across reruns); each question then takes a
few seconds (S-RAG on a sist short-answer query ~5-6 s on the project GPU). Pin
the GPU with RAG_DEVICE (default cuda:0) or CUDA_VISIBLE_DEVICES. All inference
is local / open-weight — no commercial LLM API.
"""

from __future__ import annotations

import os

import streamlit as st

from rag import config
from rag.infer import SistRAG

st.set_page_config(page_title="SIST · S-RAG 问答", layout="wide")


@st.cache_resource(
    show_spinner="首次加载模型与向量索引(约 2–4 分钟,仅一次;加载完成后会自动出现输入框)…"
)
def load_engine(device: str, smoke: bool) -> SistRAG:
    """Load the retriever + cached index + generator once, reused across reruns."""
    return SistRAG(device=device, smoke=smoke)


with st.sidebar:
    st.header("设置")
    device = st.text_input("device", os.environ.get("RAG_DEVICE", "cuda:0"))
    smoke = st.checkbox("smoke 小模型(仅调试)", value=False)
    st.divider()
    top_n = st.slider("候选池 top-N(检索)", 20, 400, config.TOP_N, step=20)
    budget = st.slider("token 预算 budget", 512, 8192, config.DEFAULT_BUDGET, step=512)
    max_new = st.slider(
        "max_new_tokens(生成上限)", 8, 512, 256, step=8
    )
    short = st.checkbox(
        "短答案抽取模式", value=True, help="关掉则让模型综合作答(长答案)"
    )
    st.caption("选择策略固定为 S-RAG(submodular-RAG)。")

engine = load_engine(device, smoke)

st.title("SIST 知识库问答 · S-RAG")
st.caption(
    "基于本地 sist 语料(22,754 段)+ S-RAG 选段 + 本地 Llama-3 生成,"
    "全程本地、无商用 API。"
)

question = st.text_input(
    "请输入问题", placeholder="例如:信息科学与技术学院有哪些研究中心?"
)
go = st.button("提问", type="primary")

if go and question.strip():
    with st.spinner("检索 → S-RAG 选段 → 生成中…"):
        res = engine.answer(
            question.strip(),
            method="srag",
            top_n=top_n,
            budget=budget,
            max_new_tokens=max_new,
            short=short,
        )

    st.subheader("回答")
    st.success(res.answer or "(空)")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("候选段 N", res.n_candidates)
    c2.metric("选中段数", res.n_selected)
    c3.metric("上下文 tokens", f"{res.cost_tokens}/{res.budget}")
    c4.metric("耗时", f"{res.latency_ms / 1000:.2f}s")

    st.subheader(f"证据段落(S-RAG 选中 {res.n_selected} 段)")
    if not res.evidence:
        st.info("本次未选中任何段落。")
    for i, e in enumerate(res.evidence, 1):
        with st.expander(
            f"[{i}] {e.title or e.pid} · score={e.score:.3f} · {e.tokens} tok"
        ):
            meta = " · ".join(x for x in [e.category, e.url] if x)
            if meta:
                st.caption(meta)
            st.write(e.text)
elif go:
    st.warning("请先输入问题。")
