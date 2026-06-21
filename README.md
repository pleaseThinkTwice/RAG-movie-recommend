# RAG 电影推荐系统

基于 RAG (Retrieval-Augmented Generation) 的电影推荐系统。

## 项目结构

```
movie_recommend/
├── src/
│   ├── core/          # 核心配置与数据模型
│   ├── data/          # 数据源
│   ├── chunking/      # 文本切分
│   ├── indexing/      # 向量索引
│   ├── query/         # Query 结构化解析
│   ├── retrieval/     # 召回层（多路并行 + RRF 融合）
│   ├── ranking/       # 精排层
│   ├── generation/    # 生成层（基于检索的解释）
│   ├── evaluation/    # 评估体系
│   └── cli/           # 命令行工具
├── config/            # 配置文件
├── data/              # 数据目录
├── notebooks/         # Jupyter Notebooks
├── tests/             # 测试
└── pyproject.toml     # 项目配置
```

## 快速开始

### 环境要求

- Python >= 3.11

### 安装

```bash
pip install -e .
```

### 开发依赖

```bash
pip install -e ".[dev]"
```

## 技术栈

- **向量数据库**: ChromaDB
- **Embedding**: sentence-transformers
- **稀疏检索**: BM25 (rank-bm25)
- **中文分词**: jieba
- **LLM**: OpenAI API
- **数据验证**: Pydantic v2
- **配置**: PyYAML
- **评估**: datasets (HuggingFace)

## License

MIT
