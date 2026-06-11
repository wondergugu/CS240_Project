# S-RAG 文档选择算法复现

本项目实现论文 *Optimal Document Selection in RAG via Combinatorial
Optimization: A Theoretical Framework* 中的文档选择核心思想：在
固定 token 预算内，从候选文档池中选择一组与查询相关且信息互补的
文档。

项目专注于 **RAG 的上下文选择阶段**，不是完整的 RAG 产品系统。
本地演示没有真实检索 embedding、prompt 拼装、LLM 生成或论文真实
数据集上的下游答案评测。

## 算法流程

```text
候选文档 D、检索相关性 rel(d, q)、token 成本 c(d)
        |
        v
对文档抽取名词/动词 lemma
        |
        v
使用 Top-L 检索文档定义概念全集 U
        |
        v
为所有 u in U 构造权重 w(u)
        |
        v
在预算 B 下选择文档集合 S
        |
        v
以加权概念覆盖 f(S) 比较选择结果
```

论文代理目标在代码中表示为：

```text
w(u) = max rel(d, q), for d in D and u in U(d)
f(S) = sum w(u), for u covered by at least one document in S
```

文档重复覆盖已有概念时，不会重复增加 `f(S)`；因此该目标自然倾向
于覆盖更多新增信息。

## 项目结构

```text
.
├── 10185_Optimal_Document_Selecti.pdf    # 项目灵感来源论文
├── requirements.txt                      # 固定依赖（含 rag/ 评测；streamlit 见隔离说明）
├── requirements-streamlit.txt            # 交互 demo 专用，仅装进隔离 venv
├── setup_nltk.py                          # 校验并修复 NLTK 数据资源
├── record.md                              # 诊断、修复与重构记录
├── srag/                                  # 核心算法包（文档选择）
│   ├── __init__.py                        # 公共包导出
│   ├── types.py                           # 数据模型与输入契约
│   ├── utils.py                           # 成本及输入校验工具
│   ├── concepts.py                        # 概念抽取与权重构建
│   ├── objective.py                       # 加权覆盖目标函数
│   └── selectors/
│       ├── __init__.py                    # 选择器公共导出
│       ├── topk.py                        # Top-k 基线
│       ├── greedy_relcost.py              # Rel/Cost 基线
│       ├── mmr.py                         # MMR 基线
│       └── srag_selector.py               # 快速版与理论版 S-RAG
├── demo/                                  # srag 选择端对照实验（ELI5/HotpotQA）
│   ├── __init__.py
│   ├── dataset_loader.py                  # 加载 ELI5/HotpotQA 为候选文档池
│   ├── mock_data.py                       # 旧合成数据池（已废弃，未被引用）
│   └── run_demo.py                        # 算法比较与小实例核验
├── rag/                                   # 端到端 RAG 评测（检索→选择→生成→评测）
│   ├── config.py                          # 模型/数据集/预算等中心配置
│   ├── retriever.py                       # bge 稠密检索封装
│   ├── corpus.py                          # KILT Wikipedia 检索语料构建
│   ├── selection.py                       # 检索候选→srag 选择器适配（token 预算）
│   ├── generator.py                       # 本地 Llama-3-8B / 冒烟生成模型
│   ├── pipeline.py                        # 端到端评测主流程
│   ├── metrics.py                         # EM/F1/ROUGE-L/BERTScore
│   ├── judge.py                           # 开源成对 LLM judge（W/T/L，去偏）
│   ├── recorder.py                        # 逐 run 产物写入 ckpt/
│   ├── export_results.py                  # 逐条结果 CSV 导出
│   ├── aggregate.py                       # 汇总 Table 1 / Table 2
│   ├── infer.py                           # 交互式 S-RAG 推理后端（SistRAG）
│   └── preprocess/                        # sist 语料清洗 + silver QA 生成、eli5 预处理
├── app.py                                 # Streamlit 交互问答前端（sist · S-RAG）
├── run_app.sh                             # 在隔离 venv 启动 app.py（详见交互 demo 节）
├── run.sh                                 # 端到端评测统一脚本（建语料/silver QA + 扫参）
├── run_rag.py                             # 端到端评测 CLI
├── corpus/                                # 检索语料（gitignore，独立于 data/）
└── tests/                                 # 模块、算法与端到端评测测试
```

### 为什么保留 `__init__.py`

`srag/__init__.py`、`srag/selectors/__init__.py` 和
`demo/__init__.py` 是标准 Python 包文件。它们使以下导入与模块执行
方式具有明确、稳定的包边界：

```python
from srag.selectors import SRAGSelector
```

```bash
python3 -m demo.run_demo
```

因此，`__init__.py` 不是无效生成文件，也不是不合规结构。本项目仅
清理 `__pycache__`、`.pytest_cache`、`.DS_Store` 等运行产物，并
通过 `.gitignore` 防止这些内容再次污染版本控制。

## 核心模块

### 数据模型与工具

文件：[srag/types.py](srag/types.py)、[srag/utils.py](srag/utils.py)

- `Document`：候选文档，持有 `doc_id`、文本、相关性
  `relevance_score` 与成本 `token_count`。
- `Query`：查询文本与可选查询 ID。
- `ConceptUniverse`：概念全集、概念权重以及文档到概念集合的映射。
- `total_cost()`：计算选中文档集合的总 token 成本。
- `validate_budget()`：拒绝负预算。
- `validate_candidates()`：拒绝重复文档 ID。

数据模型执行以下约束：

- 文档 ID 不得为空。
- 检索相关性必须为有限非负数。
- 文档 token 成本必须为正整数。
- 每个概念必须有对应非负权重。
- 文档不得覆盖概念全集以外的概念。

这些约束对应密度贪心和非负单调覆盖目标成立的前提。

### 概念抽取与概念空间

文件：[srag/concepts.py](srag/concepts.py)

该模块对应论文 Appendix B.2：

1. 使用 NLTK 分词与词性标注。
2. 仅保留名词和动词。
3. 去除停用词、非字母 token 和过短词。
4. 使用 WordNet 词形还原，如 `dogs -> dog`、`running -> run`。
5. 使用相关性最高的 Top-`L` 文档中的概念定义概念全集 `U`。
6. 将候选集合中的所有文档映射到 `U`。
7. 在整个候选集合 `D` 上按最高相关性构造每个概念的权重。

实现中每篇文档只做一次概念抽取，之后复用结果进行概念全集、
文档映射和权重构造。

### 加权覆盖目标

文件：[srag/objective.py](srag/objective.py)

`WeightedCoverageObjective` 实现论文的覆盖代理目标：

- `evaluate(selected)`：返回文档集合的加权概念覆盖得分。
- `covered_concepts(selected)`：返回当前覆盖的概念。
- `marginal_gain(doc, covered)`：返回加入一篇文档后产生的新覆盖
  价值。

实现使用 `math.fsum()` 聚合浮点权重，以避免集合遍历顺序引起的
微小舍入误差干扰单调性测试。

## 文档选择算法

文件目录：[srag/selectors](srag/selectors)

所有选择器均遵守 token 预算，并使用确定性的文档 ID 作为平局处理
依据。

### Top-k

[srag/selectors/topk.py](srag/selectors/topk.py)

按检索相关性从高到低装入可容纳的文档。它不主动避免内容冗余。

### Greedy Rel/Cost

[srag/selectors/greedy_relcost.py](srag/selectors/greedy_relcost.py)

按照以下值贪心选择：

```text
rel(d, q) / c(d)
```

该方法考虑文档成本，但不直接优化新增概念覆盖。

### MMR

[srag/selectors/mmr.py](srag/selectors/mmr.py)

MMR 通过相关性与已选文档相似度之间的权衡选文档：

```text
lambda * rel(d, q) - (1 - lambda) * redundancy(d, S)
```

`MMRSelector` 支持注入文档相似度函数。由于本地 demo 没有 embedding
检索器，默认使用概念集合的 Jaccard 相似度作为代理，并在演示输出
中标记为 `MMR (concept proxy)`。

### S-RAG Fast

[srag/selectors/srag_selector.py](srag/selectors/srag_selector.py)

`SRAGSelector.select()` 是实践用快速入口，按以下密度逐步选文档：

```text
Delta_f(d | S) / c(d)
```

默认实现使用 lazy marginal evaluation：利用子模函数的边际收益只
会下降这一性质，用堆中已有密度作为上界，仅在必要时重新计算候选
项的当前边际收益。

### S-RAG Algorithm 1

`SRAGSelector.select_theoretical()` 实现论文中的部分枚举理论入口：

1. 在所有成本可行、大小不超过 `2` 的集合中选择最佳初始答案。
2. 枚举每个成本可行、大小恰好为 `3` 的种子集合。
3. 对每个种子用边际覆盖密度贪心补全。
4. 返回全部候选方案中目标函数值最大的集合。

`select_full()` 保留为该理论入口的兼容别名。

论文中的 `(1 - 1/e)` 近似保证应仅归因于这一理论算法对
**非负单调子模代理目标** 的优化；它不等同于最终回答质量保证，也
不直接适用于快速入口 `select()`。

当剩余文档都没有新增概念覆盖时，S-RAG 不再用零收益文档填充预算，
因为它们不会改变代理目标值。

## 演示程序

文件：[demo/dataset_loader.py](demo/dataset_loader.py)、
[demo/run_demo.py](demo/run_demo.py)

`dataset_loader.py` 把 ELI5 / HotpotQA 的 Parquet 转换成统一的
`Query + 候选 Document 池`：

- HotpotQA：取 `context` 中每个段落为候选，`supporting_facts`
  命中的段落标记为参考文档（gold）。
- ELI5（KILT）：把 `output` 中的答案与 provenance 文本切成约
  140 token 的片段作为候选。
- 文档相关性为查询-文档词法启发式：查询词重叠率 + 排名先验 +
  gold 加分，**不使用 embedding 检索器**。
- `token_count` 采用空格分词计数近似。

运行前需先下载数据（见后文《论文数据集下载》）。若 `data/` 下缺少
对应 Parquet，`run_demo.py` 会报告缺失的数据集并退出。

`run_demo.py` 执行两类实验：

1. 对每个数据集采样查询，在多个 token 预算下（默认 `1024`、
   `2048`、`4096`）比较 Top-k、MMR（概念代理）、Greedy Rel/Cost
   和 `S-RAG (fast)`，输出 f(S)、覆盖概念数、参考文档召回、预算
   用量、文档数与耗时。
2. 在一个可穷举的小规模实例中，比较快速入口与
   `S-RAG (Algorithm 1)` 相对于穷举最优的近似比，并核验
   `(1 - 1/e)` 下界。

### 演示输出的指标含义

本项目不训练任何模型，下表是**选择端代理指标**：对确定性选择算法
在带标注数据集上选出的文档子集 `S` 做集合/数值统计，而非模型预测
结果。每个预算下、每种选择器输出一行。

| 指标 | 含义 | 计算方式 | 方向 |
| --- | --- | --- | --- |
| `f(S)` | 加权概念覆盖值，论文的代理目标 | 覆盖概念的权重之和 `Σ w(u)`，重复覆盖不加分 | 越高越好 |
| `Concepts` | `S` 覆盖的不同概念数量 | 覆盖概念集合的大小 | 越高越好 |
| `RefRecall` | 选回了多少数据集标注的参考文档 | `\|S ∩ gold\| / \|gold\|`，gold 来自 HotpotQA 的 `supporting_facts` 或 ELI5 的标准答案，非模型预测 | 越高越好 |
| `Budget` | 实际用掉的 token 成本 `c(S)` | 选中文档 `token_count` 之和，受预算 `B` 约束 | ≤ B 即可 |
| `Docs` | 选中的文档篇数 | `\|S\|` | 中性 |
| `Latency` | 该选择器单次选择耗时 | 选择调用的墙钟时间（毫秒） | 越低越好 |

`RefRecall` 之所以无需训练：「参考文档（gold）」是数据集自带的人工
标注，「选中文档 `S`」由确定性算法挑选，该指标只是两个集合求交后
归一化，全程不涉及任何模型推理。

小实例核验额外报告近似比 `f(Algorithm 1) / f(穷举最优)`，并核验其
不低于 `(1 - 1/e) ≈ 0.6321`，用于验证 Algorithm 1 的实现满足论文的
理论保证；该项保证只针对非负单调子模代理目标，不等于下游回答质量。

## 测试内容

**核心算法（srag/）**

| 测试文件 | 验证内容 |
| --- | --- |
| [tests/test_types.py](tests/test_types.py) | 数据模型约束与概念空间一致性 |
| [tests/test_concepts.py](tests/test_concepts.py) | 分词概念抽取、词形还原、Top-L 概念定义、单次抽取与输入校验 |
| [tests/test_objective.py](tests/test_objective.py) | 非负性、单调性、子模性与边际收益一致性 |
| [tests/test_selectors.py](tests/test_selectors.py) | 预算约束、参数校验、lazy/naive 一致性、Algorithm 1 枚举行为及冗余场景表现 |

**端到端评测与 sist（rag/）**

| 测试文件 | 验证内容 |
| --- | --- |
| [tests/test_rag_selection.py](tests/test_rag_selection.py) | 检索→选择适配器在 token 预算下的策略行为（无需模型权重） |
| [tests/test_rag_metrics.py](tests/test_rag_metrics.py) | 答案质量指标 EM/F1/ROUGE-L/BERTScore（无需模型权重） |
| [tests/test_rag_recorder.py](tests/test_rag_recorder.py) | 逐 run 产物记录（`ckpt/<model>/<run-id>/` 目录与配置） |
| [tests/test_rag_export.py](tests/test_rag_export.py) | `test_results.csv` 逐条交付物导出与合并语义 |
| [tests/test_corpus_cache.py](tests/test_corpus_cache.py) | `TorchCorpusIndex` 段落 embedding 的磁盘缓存读写与复用 |
| [tests/test_sist_corpus.py](tests/test_sist_corpus.py) | sist 语料构建（`data/sist` 分块 → `passages.jsonl`） |
| [tests/test_sist_qa.py](tests/test_sist_qa.py) | sist silver QA 加载（`qa_silver.jsonl` → `QAExample`） |
| [tests/test_sist_silver.py](tests/test_sist_silver.py) | sist silver QA 生成：解析、grounding 与池构建 |

## 依赖与 NLTK 数据

安装固定依赖：

```bash
python3 -m pip install -r requirements.txt
```

初始化或校验 NLTK 资源：

```bash
python3 setup_nltk.py
```

`setup_nltk.py` 会检查以下资源，发现缺失或损坏时仅强制重新下载该
资源，并在下载后验证：

```text
punkt_tab
averaged_perceptron_tagger_eng
wordnet
stopwords
```

默认情况下，NLTK 可能把数据放在当前用户的 `~/nltk_data` 中。例如
当前以 `root` 用户运行时，常见路径为 `/root/nltk_data`。

## 论文数据集下载

本项目使用 ELI5 和 HotpotQA 两个数据集。项目约定把参考
数据放在 `data/` 下，并按数据集名称建立一级目录：

```text
data/
├── eli5/
└── hotpotqa/
```

下载不需要新增 `requirements.txt` 依赖；下面命令只使用系统已有的
`curl`。参数 `-C -` 表示断点续传。

### ELI5

来源说明：

- 原始数据集说明：<https://huggingface.co/datasets/defunct-datasets/eli5>
- 为避免新增 `datasets` 依赖，实际下载使用公开 Parquet 文件：
  <https://huggingface.co/datasets/kilt_tasks/tree/main/eli5>

```bash
mkdir -p data/eli5

curl -L -C - \
  -o data/eli5/train-00000-of-00002.parquet \
  https://huggingface.co/datasets/kilt_tasks/resolve/main/eli5/train-00000-of-00002.parquet
curl -L -C - \
  -o data/eli5/train-00001-of-00002.parquet \
  https://huggingface.co/datasets/kilt_tasks/resolve/main/eli5/train-00001-of-00002.parquet
curl -L -C - \
  -o data/eli5/validation-00000-of-00001.parquet \
  https://huggingface.co/datasets/kilt_tasks/resolve/main/eli5/validation-00000-of-00001.parquet
curl -L -C - \
  -o data/eli5/test-00000-of-00001.parquet \
  https://huggingface.co/datasets/kilt_tasks/resolve/main/eli5/test-00000-of-00001.parquet
```

### HotpotQA

来源说明：

- 官方说明与原始 JSON 下载入口：<https://hotpotqa.github.io/>
- 实际可直接下载的公开 Parquet 镜像：
  <https://huggingface.co/datasets/hotpotqa/hotpot_qa>

```bash
mkdir -p data/hotpotqa/distractor data/hotpotqa/fullwiki

curl -L -C - \
  -o data/hotpotqa/distractor/train-00000-of-00002.parquet \
  https://huggingface.co/datasets/hotpotqa/hotpot_qa/resolve/main/distractor/train-00000-of-00002.parquet
curl -L -C - \
  -o data/hotpotqa/distractor/train-00001-of-00002.parquet \
  https://huggingface.co/datasets/hotpotqa/hotpot_qa/resolve/main/distractor/train-00001-of-00002.parquet
curl -L -C - \
  -o data/hotpotqa/distractor/validation-00000-of-00001.parquet \
  https://huggingface.co/datasets/hotpotqa/hotpot_qa/resolve/main/distractor/validation-00000-of-00001.parquet

curl -L -C - \
  -o data/hotpotqa/fullwiki/train-00000-of-00002.parquet \
  https://huggingface.co/datasets/hotpotqa/hotpot_qa/resolve/main/fullwiki/train-00000-of-00002.parquet
curl -L -C - \
  -o data/hotpotqa/fullwiki/train-00001-of-00002.parquet \
  https://huggingface.co/datasets/hotpotqa/hotpot_qa/resolve/main/fullwiki/train-00001-of-00002.parquet
curl -L -C - \
  -o data/hotpotqa/fullwiki/validation-00000-of-00001.parquet \
  https://huggingface.co/datasets/hotpotqa/hotpot_qa/resolve/main/fullwiki/validation-00000-of-00001.parquet
curl -L -C - \
  -o data/hotpotqa/fullwiki/test-00000-of-00001.parquet \
  https://huggingface.co/datasets/hotpotqa/hotpot_qa/resolve/main/fullwiki/test-00000-of-00001.parquet
```


## 运行与验证

运行完整测试：

```bash
python3 -m pytest tests
```

在真实数据集上运行选择算法对照实验（需先按上一节下载 `data/`）：

```bash
python3 -m demo.run_demo
python3 -m demo.run_demo --sample-size 20 --budgets 1024 2048
```

若希望验证过程不生成 Python 字节码或 pytest 缓存文件，可使用：

```bash
python3 -B -m pytest -p no:cacheprovider tests
python3 -B -m demo.run_demo
```

## 实现边界

本项目当前对齐的是论文的文档选择代理目标与算法结构，但不是论文
完整实验栈：

- 不包含真实检索器或 embedding 文档相似度模型。
- 不包含 LLM 生成、答案准确率或论文数据集评测。
- demo 的文档来自真实 ELI5/HotpotQA，但相关性分数为查询-文档
  词法启发式（词重叠 + 排名先验 + gold 加分），不是稠密检索器打分。
- demo 的 token 成本为单词计数近似，不是特定模型 tokenizer 结果。
- MMR 的默认相似度为概念集合 Jaccard 代理，不是 embedding 相似度。

如需复现论文完整实验结果，需要进一步接入论文对应的数据集、检索
分数、相似度表示、生成模型和评测流程。

## 端到端 RAG 评测（rag/）

在文档选择核心之上，`rag/` 提供端到端 RAG 评测，按论文「检索→选择
→生成→评测」对比 S-RAG / MMR / Top-k / Greedy 在**最终答案质量**上
的差异。**全部模型本地开源、零商业 LLM API。**

### 模型（自托管开源）

| 角色 | 论文集 | 冒烟集（快速验证） |
| --- | --- | --- |
| 检索 | `BAAI/bge-large-en-v1.5` | `BAAI/bge-small-en-v1.5` |
| 生成 | `NousResearch/Meta-Llama-3-8B-Instruct`（免门禁镜像，权重同 Meta 官方） | `Qwen/Qwen2.5-0.5B-Instruct` |
| 评判 | `prometheus-eval/prometheus-7b-v2.0` | （同生成模型占位） |
| BERTScore | `roberta-large` | `distilbert-base-uncased` |

### 语料与数据（分离）

- 检索语料 `corpus/`（**独立于 data/**，gitignore）：ELI5 用 KILT
  Wikipedia（`facebook/kilt_wikipedia` 流式构建）。
- QA `data/`：仅提供问题 + 参考答案。

### 运行

```bash
python3 -m pip install -r requirements.txt

# 统一脚本（变量控制策略/数据集/样本/预算；DATASET 默认 "eli5 sist"）
SMOKE=1 NUM_SAMPLES=5 METHODS="topk mmr greedy srag" ./run.sh
SMOKE=1 DATASET=sist NUM_SAMPLES=5 ./run.sh   # 只跑 sist（会先建 silver QA 池）

# 或直接用 CLI
python3 run_rag.py --smoke --dataset eli5 --num-samples 5 \
  --methods topk mmr greedy srag --judge

# 汇总 Table 1（指标）/ Table 2（W/T/L）
python3 -m rag.aggregate
```

`--smoke` 用小模型快速验证；去掉则用论文模型。

### 指标

短答案 EM/F1；长答案（ELI5）ROUGE-L + BERTScore；全部：开源 judge
成对 **W/T/L%**（顺序交换去偏）。

### 运行产物

`ckpt/<生成模型>/<run-id>/`：`config.json`/`config.yaml`（启动参数）
+ `results.csv`（逐条）+ `metrics_summary.csv` + `judge_wtl.csv`
（+ `best_params.json`）。论文无梯度训练，故一般无权重；「最优」对应
可调超参（如 MMR λ）。

### 现状

- **论文模型已全部本地缓存**（`bge-large-en-v1.5` + `Meta-Llama-3-8B-Instruct`
  fp32 + `prometheus-7b-v2.0` + `roberta-large`，存于 `./.hf_cache`），可设
  `HF_HUB_OFFLINE=1` 离线加载，不再受网络下载所限。
- **ELI5** 在冒烟与论文模型上均可端到端运行（长答案 ROUGE-L + BERTScore）。
- **sist** 已端到端跑通：先用本地模型读 sist 语料生成 silver QA 池
  （`rag/preprocess/sist_silver.py` + `sist_clean.py`，默认 200 条），再走
  检索 → S-RAG/MMR/Top-k/Greedy 选段 → 本地生成 → 评测（短答案 EM/F1），
  并配套交互式 demo（见下节 `app.py`）。
- **HotpotQA** 的数据管线与配置已就绪（`DATASETS` 含 `hotpotqa`，短答案
  EM/F1），可按需采样评测。

## 交互式问答演示（app.py · Streamlit · S-RAG）

在 **sist 语料**上自由提问：走「检索 top-N → **S-RAG** 选段（预算
`4096`）→ 本地 `Llama-3-8B-Instruct` 生成」，前端展示**答案 + 证据段落
（可展开看原文/出处）+ 效率指标**。后端 `rag/infer.py`（`SistRAG`）与
`rag/pipeline.evaluate` **同一条检索→选择→生成路径**，保证 demo 与评测
口径一致。全程本地、零商业 LLM API。

### 启动

```bash
bash run_app.sh                 # 默认 device cuda:0，服务在 localhost:8501
```

首次约 **2–4 分钟**加载模型（`bge-large-en-v1.5` + `Llama-3-8B` fp32，
**仅一次**，之后常驻；每次提问约 1.5–11 秒）。离线加载（`HF_HUB_OFFLINE=1`，
`run_app.sh` 已默认开启）可避免联网 HEAD 检查导致的卡顿。

命令行 / REPL（走 base 环境，**不需 venv**）：

```bash
python3 -m rag.infer "信息科学与技术学院有哪些研究中心?"
python3 -m rag.infer            # 进入交互模式
```

### 远程 SSH 查看（服务器无显示器）

Streamlit 服务在**服务器的 `localhost:8501`**，本地浏览器经 SSH 隧道访问：

1. 服务器上（你的 SSH 会话里）：`bash run_app.sh`
2. **本地机器另开一个终端**建隧道（填你平时 ssh 的 `user@host`）：

   ```bash
   ssh -N -L 8501:localhost:8501 <user>@<this-server>
   ```

   经跳板机再加 `-J <jump-user>@<jump-host>`；保持该终端开着。
3. 本地浏览器打开 `http://localhost:8501`

> 用 **VS Code Remote-SSH** 会自动转发 8501，直接点弹出的链接即可，无需手动建隧道。

### 端口与 GPU

- **换端口**：`STREAMLIT_PORT=8600 bash run_app.sh`（本地隧道同步换成 8600）；
  监听地址用 `STREAMLIT_ADDRESS` 覆盖（默认 `localhost`）。
- **换 GPU**：`RAG_DEVICE=cuda:1 bash run_app.sh`（本机 2 张卡；`cuda:0` 在忙就
  用 `cuda:1`；无 GPU 自动回退 CPU）。
- `run_app.sh` 以 `--server.headless true` 启动，**跳过首次「输入邮箱」交互提示**
  （否则在无显示器服务器上会卡住），并关闭用量统计。

### 生成长度与答案风格

- 默认 `max_new_tokens=256`（滑条最高 `512`），列表型问题不会像 `32` 那样被截断。
- 侧栏「短答案抽取模式」默认开（factoid 给最短 span）；想要成段自然语言就**关掉**它。
- 交互默认值是 `rag/infer.py` 里独立的 `DEFAULT_MAX_NEW_TOKENS`，**不改动**评测口径
  `config.MAX_NEW_TOKENS["sist"]=32`。

### 依赖隔离（重要）

`streamlit` 硬性要求 `anyio>=4`，而本环境的 `jupyter-server / jupyterlab` 要求
`anyio<4`，二者**不可共用同一 anyio 版本**。因此 **streamlit 装在独立 venv
`.venv-streamlit/`**（`run_app.sh` 首次用 `--system-site-packages` 自动创建，
**复用 base 的 torch/transformers，不重下 30GB**；venv 内 anyio≥4，base 保持
`anyio 3.7.1`，jupyter kernel 不受影响）。**切勿把 streamlit 装进 base 环境**
——会把 base 的 anyio 升到 4.x 而**破坏 Jupyter kernel**。venv 依赖见
`requirements-streamlit.txt`；`.venv-streamlit/` 是本地产物，已被 `.gitignore` 忽略。
