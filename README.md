# RAG 电影推荐系统

基于 RAG（Retrieval-Augmented Generation）的电影推荐系统。接收自然语言查询，返回 Top-K 电影及 AI 生成的 Grounded 推荐理由。

```
User Query (NL) → Query 结构化解析 (LLM) → 三路召回 (Vector + BM25 + Metadata)
                 → RRF 融合 → 精排 (Cross-Encoder) → 生成解释 (LLM)
```

## 快速开始

### 环境要求

- Python >= 3.11
- 4GB+ GPU 推荐（CPU 也可）
- DeepSeek API Key（可选，无 Key 时 Query 解析降级运行）

### 安装

```bash
pip install chromadb sentence-transformers rank-bm25 jieba openai pydantic pyyaml datasets tqdm httpx tenacity rich

# HuggingFace 下载慢？设置镜像：
export HF_ENDPOINT=https://hf-mirror.com
```

### 设置

```bash
export DEEPSEEK_API_KEY=your_api_key_here
```

### 四步运行

```bash
# 1. 数据加载与预处理（~30K 电影，1 分钟）
python -c "from src.data.pipeline import run_pipeline; run_pipeline()"

# 2. 结构化 Chunking（2 分钟）
python -c "from src.chunking.chunker import run_chunking; run_chunking()"

# 3. 构建索引 — Embedding + ChromaDB + BM25（~30 分钟，GPU）
python -m src.cli.build_index

# 4. 查询推荐
python -m src.cli.recommend "像《盗梦空间》那种烧脑科幻片，2010年以后"

# 5. 运行评估
python -m src.cli.evaluate
```

## 项目结构

```
movie_recommend/
├── config/
│   ├── default.yaml          # 所有可调参数
│   └── style_dict.json       # 风格/观感词典（50+ 标签）
├── src/
│   ├── core/                 # config.py, schemas.py (Pydantic v2)
│   ├── data/                 # HuggingFace 加载 → 清洗 → 过滤 → JSONL
│   ├── chunking/             # 3 类结构化 chunk + TF-IDF 关键词 + 风格匹配
│   ├── indexing/             # Embedder (bge-large-zh), ChromaDB, BM25
│   ├── query/                # LLM Query 解析 + Similar-to 电影匹配
│   ├── retrieval/            # 向量/BM25 检索、元数据过滤、RRF(k=60)、编排器
│   ├── ranking/              # Cross-Encoder 精排 (bge-reranker-large)
│   ├── generation/           # LLM 解释生成 + Grounding 3-gram 验证
│   ├── evaluation/           # Recall@K, NDCG@K, MRR + EvalSet + Runner
│   └── cli/                  # build_index, recommend, evaluate
├── tests/                    # 22+ tests (RRF, metrics, metadata filter)
└── data/                     # .gitignored: movies.jsonl, chunks.jsonl, indexes/
```

## 技术栈

| 组件 | 选型 | 说明 |
|------|------|------|
| Embedding | BAAI/bge-small-zh-v1.5 | 512-dim, C-MTEB 检索 ~62 |
| 向量库 | ChromaDB | HNSW, cosine, 元数据 where |
| 关键词 | BM25 (rank_bm25 + jieba) | k1=1.5, b=0.75 |
| 融合 | RRF (k=60) | Reciprocal Rank Fusion |
| 精排 | BAAI/bge-reranker-base | Cross-Encoder |
| LLM | DeepSeek-V3 (API) | Query 解析 + 解释生成 |
| 数据 | HuggingFace dataset | 30,429 部豆瓣电影 |

## 设计特点

- **三路混合召回**：向量语义 + BM25 关键词 + 元数据硬约束，互补盲区
- **RRF 融合**：只看排名不看分数，避免不同检索器分数尺度不可比
- **结构化 Chunking**：plot / themes_style / reviews 三类，各自对应不同 query 意图
- **防幻觉**：结构化字段（标题/评分/导演）模板渲染，LLM 只生成主观推荐理由
- **Grounding 验证**：3-gram 回溯检查解释中的声明能否在检索 chunk 中找到依据

## 评估

| 指标 | 说明 |
|------|------|
| Recall@K | Top-K 中相关电影占比 |
| NDCG@K | 考虑排序位置的折扣累积增益 |
| MRR | 第一个相关电影的倒数排名 |
| LLM-as-judge | 解释事实一致性评分 |

## License

MIT
