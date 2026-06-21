# RAG 电影推荐系统 — 实施计划

## 项目背景

基于 `RAG电影推荐_项目细节与面试预判.md` 中的设计文档，从零构建一个基于 RAG 的电影推荐系统。系统接收自然语言查询（如"像《盗梦空间》那种烧脑但不要太晦涩的科幻片，2010年以后的"），返回 Top-K 电影列表以及基于检索内容生成的推荐理由。

**核心架构**：Query 结构化解析 (LLM) → 三路召回 (向量 + BM25 + 元数据过滤) → RRF 融合 → 精排 (Cross-Encoder) → 生成解释 (LLM)

**核心技术栈**：Python、bge-large-zh-v1.5、ChromaDB、rank_bm25、jieba、bge-reranker-large、DeepSeek-V3 API、~5 万部电影 (TMDB + 豆瓣)

## 项目结构

```
movie_recommend/
├── pyproject.toml
├── config/default.yaml          # 所有可调参数
├── src/
│   ├── core/                    # config.py、schemas.py（Pydantic 模型）
│   ├── data/                    # TMDB 客户端、豆瓣爬虫、数据清洗、过滤器、编排管道
│   ├── chunking/                # 结构化切分器、themes_extractor（离线 LLM）
│   ├── indexing/                # embedder.py、vector_store.py（ChromaDB）、bm25_index.py
│   ├── query/                   # parser.py（LLM query 解析）、similar_to.py
│   ├── retrieval/               # vector_retriever、bm25_retriever、metadata_filter、rrf、orchestrator
│   ├── ranking/                 # reranker.py（bge-reranker-large）
│   ├── generation/              # explainer.py、grounding.py
│   ├── evaluation/              # eval_set、metrics（Recall@K、NDCG@K、MRR）、judge、runner
│   └── cli/                     # build_data、build_index、recommend、evaluate
├── tests/
├── data/                        # .gitignored：raw/、processed/movies.jsonl、indexes/
└── notebooks/                   # 数据探索、Bad Case 分析
```

## 分阶段实施计划

### Phase 0 — 项目基础（3-4 天）
**目标**：可运行的项目骨架，所有接口定义完毕，配置系统正常工作。

- `pyproject.toml` 依赖：`chromadb`、`sentence-transformers`、`rank-bm25`、`jieba`、`openai`（兼容 DeepSeek）、`pydantic`、`pyyaml`、`aiohttp`、`beautifulsoup4`、`rapidfuzz`、`tqdm`、`tenacity`
- `src/core/config.py`：YAML 配置加载器，环境变量读取 API Key
- `src/core/schemas.py`：`Movie`、`Chunk`、`ParsedQuery`、`RetrievalResult`、`Recommendation` 等 Pydantic 模型
- 所有模块的接口定义（方法签名桩）
- `tests/conftest.py`：5 部电影的小型测试数据集、Mock LLM 客户端
- CLI 入口桩（打印使用说明）

### Phase 1 — 数据管道（7-10 天）
**目标**：产出 ~5 万条经过清洗、去重的电影数据，存储为 `data/processed/movies.jsonl`。

- **TMDB 客户端**：分页拉取电影发现列表 + 详情 + 演职员信息。遵守速率限制。原始响应缓存到磁盘。
- **豆瓣爬虫**：优先使用现有公开数据集（Kaggle/GitHub）作为种子数据；缺口部分通过爬虫补全，使用礼貌的请求间隔和轮换 User-Agent。
- **跨源匹配**：TMDB 与豆瓣之间通过标题模糊匹配（rapidfuzz）+ 年份 + 导演进行关联。
- **数据清洗**：去除 HTML 标签、统一标点符号（NFKC 归一化）、折叠重复字符。
- **过滤规则**：剧情简介 ≥ 50 字、豆瓣评分人数 ≥ 100、至少有 1 条用户评论。
- **去重**：IMDB ID 为主键，豆瓣 ID 为辅键。
- **管道编排器**：支持断点续跑的异步数据管道，带进度上报。

### Phase 2 — Chunking 与主题提取（5-7 天）
**目标**：每部电影产出 3 个结构化 chunk；themes_style 字段由 LLM 离线填充。

- **结构化切分器**：剧情 chunk、主题风格 chunk（LLM 填充）、评论 chunk（Top-5）。每个 chunk 携带完整元数据。
- **主题提取器**：~5 万次 DeepSeek-V3 API 调用。Prompt：从 Top-20 评论中提取 3-5 个主题/风格/观感关键词。温度=0，JSON 模式。预计成本约 50 元人民币，5 并发约 30-60 分钟。通过 sidecar 缓存文件支持断点续跑。
- **关键风险**：在完整运行前，务必花 1 天时间在 100 个样本上迭代 prompt，人工检查质量。这一步的输出是整个下游系统的检索基础，出错代价极高。

### Phase 3 — Embedding 与向量索引（4-5 天）
**目标**：所有 chunk 完成向量化并存入 ChromaDB，支持检索。

- **Embedding**：`BAAI/bge-large-zh-v1.5`，通过 sentence-transformers 加载，GPU 推理，batch_size=32，normalize_embeddings=True。超过 512 token 的文本截断。
- **向量库**：ChromaDB PersistentClient，cosine 距离空间，元数据字段可用于过滤。元数据编码：year → `$gte`/`$lte`，genres → `$contains`。
- **构建脚本**：加载 chunk → 批量编码 → 写入 ChromaDB。约 30-40 分钟。

### Phase 4 — BM25 索引（2-3 天）
**目标**：基于 jieba 分词的 BM25 关键词检索引擎。可与 Phase 3 并行。

- `rank_bm25.BM25Okapi`（k1=1.5, b=0.75），使用 jieba 预分词。
- pickle 序列化持久化。
- **元数据集成策略**：从 BM25 取 `top_k * 2` 条结果，再用元数据约束后过滤，返回 top_k。记录过滤丢弃率以调优。

### Phase 5 — Query 结构化解析（3-4 天）
**目标**：自然语言查询 → 结构化 `ParsedQuery`。可与 Phase 2-4 并行开发。

- **LLM 解析器**：DeepSeek-V3，温度=0，JSON 模式。Prompt 提取 `{semantic_query, filters, similar_to}`。解析失败时优雅降级。
- **Similar-to 解析器**：精确匹配 + 模糊匹配电影名称 → movie_id。找到的参照电影的 themes_style embedding 将作为额外一路向量检索。
- **缓存**：Query → ParsedQuery 映射缓存，节省迭代时的 API 费用。

### Phase 6 — 召回层：三路并行 + RRF 融合（5-7 天）
**目标**：完整召回管道：ParsedQuery → Top-50 电影。**系统的核心。**

- **向量检索器**：对 semantic_query 做 embedding，查询 ChromaDB（带元数据 where 前置过滤）。
- **BM25 检索器**：分词、搜索、元数据后置过滤。
- **元数据过滤器**：FilterConstraints → ChromaDB where 字典。
- **RRF**：纯函数，k=60。对每条检索路径计算 1/(k+rank) 之和。先按 movie_id 去重（保留最高排名）再做 RRF。
- **编排器**：通过 asyncio.gather 并行执行向量检索 + BM25 检索（+ 可选的 similar_to 检索），RRF 融合，返回 Top-50。
- **兜底策略**：如果前置过滤后结果 < 10 条，放宽约束软降级。

### Phase 7 — 精排层（3-4 天）
**目标**：Cross-Encoder 重排序，Top-50 → Top-10。

- `BAAI/bge-reranker-large`，通过 sentence-transformers CrossEncoder 加载，GPU 推理。
- 长 chunk 截断策略：`"{标题} | {类型} | {前400字}"`。
- 50 个候选对批量推理。目标延迟：<300ms。

### Phase 8 — 生成层（4-5 天）
**目标**：为每部推荐电影生成 2-3 句有据可依的推荐理由。

- **解释生成器**：DeepSeek-V3，温度=0.3。Prompt 包含用户 query + 全部 3 个 chunk。严格的 Grounding 约束（"不要编造"、"无法确认时请明确说明"）。
- **结构化字段渲染**：标题、年份、评分、导演、类型等结构化字段由模板直接填充——绝不让 LLM 生成（杜绝事实幻觉）。
- **输出格式**：结构化头部 + AI 生成的推荐理由。
- **并发**：10 部电影的解释并行生成，总延迟约 1.5 秒。

### Phase 9 — 评估体系（5-7 天）
**目标**：可复现的指标。基线数据对标设计文档的 v1→v2→v3 演进路径。

- **评估集**：200 条标注 query。1/3 手写、1/3 改编自豆瓣片单、1/3 LLM 生成。每条 query 标注 5-15 部相关电影。
- **指标**：`recall_at_k`、`ndcg_at_k`、`mrr`——纯函数，完整的单元测试。同时追踪 Recall@50（召回后）和 Recall@10（精排后）。
- **LLM-as-judge**：用 Claude 做裁判（不同模型家族，避免偏差）。抽样 50 条解释，从事实一致性、需求相关性两个维度评 1-5 分。
- **评估运行器**：完整的评估管道。结果带 git hash 和配置快照缓存，支持版本对比。
- **关键提醒**：设计文档作者的首要教训——在 Phase 1 结束时就构建 50 条 query 的评估子集，不要等到 Phase 9。每次改动都有数字反馈。

### Phase 10 — 迭代与打磨（5-10 天）
**目标**：优化指标、修复 Bad Case、准备展示。

- **Bad Case 分析**：将失败归因分类（召回遗漏 vs. 排序靠后 vs. 幻觉生成）。
- **参数扫描**：RRF k ∈ {30, 60, 100}，各路召回数量 ∈ {20, 30, 50, 100}。
- **Prompt 优化**：基于 LLM-judge 反馈迭代解释生成的 prompt。
- **打磨**：CLI 用户体验、README 含架构图、快速启动指南。可选：Streamlit/Gradio Web 演示界面。

## 依赖关系图

```
Phase 0 → Phase 1 → Phase 2 → Phase 3 ↘
                              → Phase 4 → Phase 6 → Phase 7 → Phase 8 → Phase 9 → Phase 10
                    Phase 5 ↗
```
Phase 5 可与 Phase 2-4 并行开发。

## 预估时间线

| 阶段 | 天数 | 累计 |
|------|------|------|
| Phase 0 - 项目基础 | 3-4 | 4 |
| Phase 1 - 数据管道 | 7-10 | 14 |
| Phase 2 - Chunking 与主题提取 | 5-7 | 21 |
| Phase 3 - Embedding 与向量索引 | 4-5 | 26 |
| Phase 4 - BM25 索引 | 2-3 | 29 |
| Phase 5 - Query 解析 | 3-4 | 33（可并行） |
| Phase 6 - 召回层 + RRF | 5-7 | 40 |
| Phase 7 - 精排层 | 3-4 | 44 |
| Phase 8 - 生成层 | 4-5 | 49 |
| Phase 9 - 评估体系 | 5-7 | 56 |
| Phase 10 - 迭代打磨 | 5-10 | 66 |

**总计：约 8-10 周**（业余时间，晚上/周末推进）。全职投入约 4-5 周。

## 启动前的关键决策

1. **Python 3.11**（sentence-transformers 和 chromadb 的最佳兼容版本）
2. **`uv`** 管理依赖（安装快、PyTorch 生态兼容好）
3. **豆瓣数据**：优先使用现有公开数据集（Kaggle）作为种子——这是最高风险的环节
4. **DeepSeek API**：Phase 2 前确认 API 可用性和速率限制（5 万次主题提取调用）
5. **GPU 验证**：Phase 3 前确认 4090 能正常加载 bge-large-zh-v1.5 和 bge-reranker-large
6. **评估集早建**：Phase 1 就建 50 条 query 子集——不要拖到 Phase 9

## 验证方案

每个阶段边界都有对应的测试（详见各阶段详情）。端到端验证：

- `python -m src.cli.recommend "像《盗梦空间》那种烧脑科幻，2010年以后"` → 输出带解释的排序推荐列表
- `python -m src.cli.evaluate` → 输出 Recall@10 ≈ 0.73，NDCG@10 ≈ 0.71（对标设计文档 v3 指标）
- `pytest` → 所有测试通过
- README 能让一个陌生人在 10 分钟内跑起系统
