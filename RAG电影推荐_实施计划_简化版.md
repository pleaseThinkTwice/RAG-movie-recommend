# RAG 电影推荐系统 — 实施计划（简化版）

> **核心思路**：用现成数据集替代爬虫、用轻量模型替代大模型、用关键词提取替代大规模 LLM 调用。保持架构完整（三路召回 + RRF + 精排 + 生成），但在每个环节选择更省时省钱的方案。**麻雀虽小，五脏俱全。**

---

## 一、与原计划的关键差异

| 环节 | 原计划 | 简化版 | 节省 |
|------|--------|--------|------|
| **数据来源** | 从零爬 TMDB + 豆瓣，5 万部 | HuggingFace 现成数据集，5-8 千部 | **省 5-7 天，零爬虫风险** |
| **Embedding** | bge-large-zh-v1.5 (1024维，需 GPU) | bge-small-zh-v1.5 (512维，CPU 可跑) | **省 GPU 成本，存储减半** |
| **主题提取** | 5 万次 DeepSeek-V3 调用（~50 元） | jieba TF-IDF 关键词 + 数据集已有标签 | **省 50 元 + 1-2 天** |
| **精排模型** | bge-reranker-large | bge-reranker-base（或 v1 直接跳过） | **更轻量，CPU 可跑** |
| **评估集** | 200 条标注 query | 50-80 条标注 query | **省 2-3 天标注量** |
| **数据规模** | ~5 万部电影，15 万 chunk | ~5-8 千部电影，1.5-2.4 万 chunk | **全部索引在内存** |
| **总时长** | 8-10 周（业余） | **3-5 周（业余）** | **减半以上** |

---

## 二、数据来源调研结果

### 首选：HuggingFace `MangoGoes/douban_movie_info`

直接从 HuggingFace 加载，无需爬虫：

```
pip install datasets
```

```python
from datasets import load_dataset
dataset = load_dataset("MangoGoes/douban_movie_info", split="train")
```

**字段齐全**（31 列）：电影名、导演、演员、编剧、类型、上映日期、制片国家、语言、片长、IMDb ID、豆瓣评分、评分人数、评论摘要。基本覆盖了原设计文档 Schema 的所有字段。

**数据量**：数万条记录，经过质量过滤（有评分 + 有评论 + 评分人数 ≥ 100）后预计剩余 **5,000-8,000 部**。

### 备选补充

如果 HuggingFace 数据集的评论字段不够丰富，可补充：
- **OpenI 启智社区 `Douban_Movie_Short_Reviews`**：~1 万部电影，1031 万条短评，按 22 种类型分类，自带情感标签
- **GitHub `SophonPlus/ChineseNlpCorpus` (DMSC V2)**：28 部热门电影，212 万条评论（MovieLens 兼容格式），质量高但电影少

---

## 三、Embedding 模型对比与选型

### bge-small-zh-v1.5 vs bge-large-zh-v1.5

| 指标 | bge-small-zh-v1.5 ✅ | bge-large-zh-v1.5 |
|------|---------------------|-------------------|
| 向量维度 | 512 | 1024 |
| C-MTEB 检索分 | 61.77 | ~70.46 |
| 单条推理耗时 | ~38ms | ~180ms |
| 显存/内存占用 | ~480MB | ~2.8GB |
| QPS (单 GPU) | 3000-3500 | ~800 |
| 参数量 | ~110M | ~330M+ |
| **CPU 可跑** | ✅ 轻松 | ❌ 吃力 |
| **存储 (8K×3 chunk)** | ~48MB | ~96MB |

**结论**：small 的检索精度损失约 8-12%，但换来 4-5 倍推理加速、1/5 内存、CPU 可跑。对于原型项目完全够用。而且我们的三路混合召回 + 精排本身就是设计来弥补单路检索不足的——多路融合和 Reranker 可以补回很大一部分精度损失。

---

## 四、简化的主题提取方案

**原方案**：5 万次 DeepSeek-V3 调用，提取 3-5 个主题/风格关键词。

**简化方案**：不用 LLM，用轻量 NLP 方案：

1. **数据集已有类型标签**：`类型` 字段（如"剧情/喜剧/动作"），直接作为分类特征
2. **jieba TF-IDF 关键词提取**：从评论 + 剧情简介中自动抽取 Top-10 关键词
3. **手工规则**：预定义一组风格词典（烧脑、治愈、慢节奏、反转、催泪……约 50-80 个词），用文本匹配 + embedding 相似度打标

这样 themes_style chunk 的内容来源变为：
```
[类型] 剧情 / 悬疑 / 科幻
[关键词] 梦境 多层 反转 潜意识 时间
[风格标签] 烧脑 非线性叙事 视觉奇观
```

**付出的代价**：风格描述的细腻度不如 LLM 提取。**换来的收益**：省 50 元 API 费用 + 1-2 天等待时间 + 零 API 依赖风险。

> 后续如果效果不够好，可以在评估阶段针对性地对 100-200 部电影跑 LLM 做主题增强，而不是全部。

---

## 五、分阶段实施计划

### Phase 0 — 项目骨架（2-3 天）

**目标**：可运行的基础框架。

- `pyproject.toml`：精简依赖（`chromadb`、`sentence-transformers`、`rank-bm25`、`jieba`、`pydantic`、`pyyaml`、`datasets`、`tqdm`）
  - LLM 调用用 `openai` 库连 DeepSeek API（仅 query 解析 + 解释生成用）
- `src/core/`：config.py、schemas.py（Pydantic 模型，比原计划少几个字段）
- `tests/conftest.py`：5 部电影的 mock 数据
- 所有模块的接口桩

---

### Phase 1 — 数据加载与预处理（2-3 天）

**目标**：从 HuggingFace 加载数据，清洗过滤后输出 `data/processed/movies.jsonl`（5000-8000 部）。

**工作内容**：

1. **加载数据集**：`datasets.load_dataset("MangoGoes/douban_movie_info")`
2. **字段映射**：将 31 个原始字段映射到我们的 Schema
   - 年份从 `上映日期` 提取
   - `类型` 字段按 `/` 分割成列表
   - `score` → `rating_douban`，`score_amt` → `rating_count`
3. **数据清洗**：
   - 去 HTML 标签（如果有）
   - 统一标点（NFKC）
   - 折叠重复字符
4. **过滤**：
   - `comment` 字段非空且长度 ≥ 20 字
   - `score_amt` ≥ 100
   - 有 `类型` 标签
5. **输出**：`data/processed/movies.jsonl`

**不需要做的**：爬虫、TMDB 匹配、跨源去重——数据集本身就是合并好的。

---

### Phase 2 — Chunking 与主题提取（2-3 天）

**目标**：每部电影产出 3 个结构化 chunk。**不用 LLM。**

**工作内容**：

1. **结构化切分器**：
   - `plot` chunk：从 `text_info`（剧情文本）提取
   - `themes_style` chunk：组合 `类型` 标签 + TF-IDF 关键词 + 风格词典匹配结果
   - `reviews` chunk：`comment` 字段（评论摘要）
2. **关键词提取**：
   - 用 jieba 分词 + TF-IDF 对每部电影的评论/剧情文本抽 Top-10 关键词
   - 预定义风格词典（`style_dict.json`，约 50-80 个 mood/风格词），文本匹配打标
3. **输出**：`data/processed/chunks.jsonl`

---

### Phase 3 — Embedding 与向量索引（2-3 天）

**目标**：所有 chunk 用 bge-small-zh-v1.5 编码，存入 ChromaDB。

**工作内容**：

1. **Embedder**：`BAAI/bge-small-zh-v1.5`，batch_size=64（模型更小可以更大 batch），CPU 或 GPU 均可
2. **向量库**：ChromaDB PersistentClient，cosine 距离
3. **元数据编码**：year → `$gte/$lte`，genres → `$contains`
4. **构建脚本**：一次性跑完，数据量小（~2 万 chunk），CPU 上约 5-10 分钟

**验证**：embed "烧脑科幻" → 检查 Top-10 是否包含《盗梦空间》《星际穿越》等。

---

### Phase 4 — BM25 索引（1-2 天）

**目标**：jieba + BM25 关键词索引。可与 Phase 3 并行。

- `rank_bm25.BM25Okapi`（k1=1.5, b=0.75）
- jieba 预分词所有 chunk 文本
- pickle 持久化
- 数据量小，全部在内存，无需分布式

---

### Phase 5 — Query 结构化解析（2-3 天）

**目标**：NL query → `ParsedQuery`。**保留 LLM**（这一步 LLM 不可替代，但单次调用极便宜）。

- DeepSeek-V3，温度=0，JSON 模式
- Prompt 提取 `{semantic_query, filters, similar_to}`
- 解析失败降级 + 结果缓存
- **成本**：~100 次/天开发迭代 × $0.0001/次 ≈ 几乎免费

---

### Phase 6 — 召回层：三路并行 + RRF 融合（3-5 天）

**目标**：ParsedQuery → Top-50。**架构核心，跟原计划一致。**

- 向量检索 + BM25 检索 + 元数据过滤，三路并行（asyncio.gather）
- RRF 融合（k=60）
- 兜底策略：结果不足 10 条时放宽约束

---

### Phase 7 — 精排层（1-3 天）

**目标**：Top-50 → Top-10。**两种方案可选。**

| 方案 | 说明 | 适合 |
|------|------|------|
| **A（推荐）** | bge-reranker-base，比 large 轻量，CPU 可跑 | 有基本的 GPU/CPU 资源 |
| **B（最简）** | v1 跳过精排，RRF 直接输出 Top-10 | 赶时间先跑通全链路 |

- 方案 A：CrossEncoder 批量推理 50 对 → 输出 Top-10
- 方案 B：RRF 融合后直接取 Top-10，省掉一个模型依赖

---

### Phase 8 — 生成层（2-3 天）

**目标**：为 Top-10 电影生成推荐理由。**保留 LLM**（可解释性是项目的核心亮点）。

- DeepSeek-V3，温度=0.3
- Prompt 包含用户 query + 3 个 chunk
- 结构化字段（标题/评分/导演/类型）模板渲染，LLM 只生成推荐理由部分
- 10 部电影并行调用，总延迟 ~1.5s
- **成本**：每次查询约 $0.001（10 次生成调用），开发期间几乎忽略不计

---

### Phase 9 — 评估体系（2-4 天）

**目标**：可复现的评估，支撑迭代优化。

- **评估集**：50-80 条 query（原计划 200），够看出趋势即可
  - 1/3 手写（覆盖 mood/实体/混合/similar-to 四类）
  - 1/3 改编豆瓣片单
  - 1/3 LLM 生成
- **指标**：Recall@10、NDCG@10、MRR
- **LLM-as-judge**：抽样 20 条解释评估事实一致性（Claude 或 DeepSeek）
- **版本追踪**：v1（纯向量）→ v2（+BM25+元数据过滤）→ v3（+精排）的指标演进

---

### Phase 10 — 打磨（2-4 天）

**目标**：可以展示的完整系统。

- Bad case 分析 + 快速修复
- CLI 用户体验优化（彩色输出、进度条）
- README 含架构图 + 快速启动指南
- 可选：一个极简 Streamlit/Gradio Web 界面（一个 `app.py`）

---

## 六、依赖关系图

```
Phase 0 (骨架) → Phase 1 (数据加载) → Phase 2 (Chunking)
                                            ↓
                              Phase 3 (Embedding) ←→ Phase 4 (BM25)
                                            ↓
                              Phase 5 (Query解析，可并行)
                                            ↓
                              Phase 6 (召回+RRF，核心)
                                            ↓
                              Phase 7 (精排，可跳过v1)
                                            ↓
                              Phase 8 (生成解释)
                                            ↓
                              Phase 9 (评估) → Phase 10 (打磨)
```

---

## 七、预估时间线

| 阶段 | 天数 | 累计 | 关键产出 |
|------|------|------|----------|
| Phase 0 | 2-3 | 3 | 项目骨架可运行 |
| Phase 1 | 2-3 | 6 | movies.jsonl (5-8K条) |
| Phase 2 | 2-3 | 9 | chunks.jsonl (1.5-2.4万条) |
| Phase 3 | 2-3 | 12 | ChromaDB 索引可检索 |
| Phase 4 | 1-2 | 14 | BM25 索引可检索 |
| Phase 5 | 2-3 | 17 | Query 解析可用 |
| Phase 6 | 3-5 | 22 | **全链路召回打通** |
| Phase 7 | 1-3 | 25 | 精排可用（或跳过） |
| Phase 8 | 2-3 | 28 | 带解释的推荐输出 |
| Phase 9 | 2-4 | 32 | 评估指标可量化 |
| Phase 10 | 2-4 | 36 | 可演示的完整系统 |

**总计：约 5 周（业余时间）**，全职约 2-3 周。

---

## 八、成本估算

| 项目 | 原计划 | 简化版 |
|------|--------|--------|
| 数据采集 | 数天爬虫 + 可能被封 IP | 0（HuggingFace 免费） |
| Themes 提取 LLM | ~50 元（5万次 DeepSeek） | 0（jieba 关键词） |
| Embedding GPU | 需要 4090 | CPU 可跑，有 GPU 更快 |
| Query 解析 + 解释生成 | ~0.01 元/查询 | ~0.001 元/查询 |
| 评估 Judge | Claude API（抽样50条） | DeepSeek 自评或少量 Claude |
| **总 LLM API 成本** | ~60-80 元 | **< 5 元** |

---

## 九、一份具体的依赖清单

```toml
[project]
name = "movie-recommend"
requires-python = ">=3.11"
dependencies = [
    "chromadb>=0.5",           # 向量库
    "sentence-transformers>=3", # bge-small-zh-v1.5 + bge-reranker-base
    "rank-bm25>=0.2",          # BM25 检索
    "jieba>=0.42",             # 中文分词
    "openai>=1.0",             # DeepSeek API（兼容 OpenAI 接口）
    "pydantic>=2.0",           # 数据模型
    "pyyaml>=6.0",             # 配置文件
    "datasets>=2.0",           # HuggingFace 数据加载
    "tqdm>=4.0",               # 进度条
    "httpx>=0.27",             # HTTP 客户端
    "tenacity>=8.0",           # 重试机制
]
```

---

## 十、"五脏俱全"检查清单

对照原设计文档，确认每个核心组件都存在：

- [x] **Query 结构化解析**（LLM） — 保留
- [x] **向量检索**（bge-small-zh + ChromaDB） — 轻量化
- [x] **BM25 关键词检索**（rank_bm25 + jieba） — 保留
- [x] **元数据过滤**（ChromaDB where） — 保留
- [x] **RRF 融合**（k=60） — 保留，完全一致
- [x] **精排 Cross-Encoder**（bge-reranker-base） — 轻量化（或 v1 跳过）
- [x] **生成层 Grounding**（LLM + chunk 引用） — 保留
- [x] **结构化字段防幻觉**（模板渲染） — 保留
- [x] **评估体系**（Recall/NDCG/MRR + judge） — 缩减规模
- [x] **三版本指标演进**（v1→v2→v3） — 保留，版本对比逻辑不变

---

## 十一、启动前确认事项

1. [ ] HuggingFace 能正常访问，`MangoGoes/douban_movie_info` 能加载
2. [ ] `BAAI/bge-small-zh-v1.5` 能下载（国内可能需要镜像）
3. [ ] DeepSeek API Key 可用（或替代：阿里百炼/硅基流动的便宜模型）
4. [ ] Python 3.11 环境就绪
5. [ ] 目标：先跑通全链路（哪怕指标低），再迭代优化
