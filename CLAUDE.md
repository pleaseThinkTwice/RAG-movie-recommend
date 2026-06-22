# CLAUDE.md — 项目级配置与偏好

## 环境

- **Python**: `C:/python3.11/python.exe`（不是系统默认的 3.6）
- **Shell**: Git Bash（Unix 风格，`/dev/null` 不是 `NUL`）
- **GPU**: RTX 3050 Ti Laptop，4GB VRAM
- **OS**: Windows 11

## 成本控制（重要）

- **数据来源优先用现成的**：HuggingFace datasets、Kaggle、GitHub 公开数据集，不要从零爬虫
- **模型选轻量**：bge-small-zh-v1.5 而非 large，reranker-base 而非 large。4GB VRAM 跑不动大模型
- **LLM 调用能省则省**：离线批量任务（如主题提取）优先用 jieba TF-IDF 替代。Cache 所有 LLM 结果
- **HF 下载用镜像**：`HF_ENDPOINT=https://hf-mirror.com`，国内直连经常 SSL 超时
- **PyPI 用阿里云镜像**：`--index-url https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com`

## API Key 安全

- **绝不硬编码 API Key** 在任何 `.py` 文件中
- 用环境变量 `DEEPSEEK_API_KEY` 读取
- 测试脚本（含 key 的）加入 `.gitignore`
- 提交前检查：`git diff --cached | grep -i key`

## Git 工作流

- **写完一个功能就 commit + push**，不要攒着
- Commit message 用中文，描述做了什么
- 不要 commit 大文件（`data/processed/`、`data/indexes/`、`.pkl`、`.bin`）
- Push 前确保 API key 不在 staged files 里

## 代码风格

- Python 文件用 UTF-8，终端输出统一用 `io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')`
- 数据模型用 Pydantic v2
- 打印中文时处理编码（Windows 终端默认 GBK）
- 长任务（>30s）用 `run_in_background: true`

## 项目结构约定

```
movie_recommend/
├── config/          # YAML + JSON 配置
├── src/             # 全部源码
├── tests/           # pytest
├── data/            # .gitignored 大数据
├── *.md             # 文档（中文）
└── temp_*.py        # .gitignored 临时脚本
```

## 运行命令速查

```bash
# 环境变量
export DEEPSEEK_API_KEY=your_key
export HF_ENDPOINT=https://hf-mirror.com

# 安装
C:/python3.11/python.exe -m pip install <package>

# 测试
C:/python3.11/python.exe -m pytest tests/ -v

# 索引重建
C:/python3.11/python.exe -m src.cli.build_index

# 推荐查询
C:/python3.11/python.exe -m src.cli.recommend "query"
```

## 文档入口

- [项目介绍](项目介绍.md) — 对外展示用
- [实施计划（简化版）](RAG电影推荐_实施计划_简化版.md) — 开发规划
- [面试预判](RAG电影推荐_项目细节与面试预判.md) — 面试准备
- [论文](论文_RAG电影推荐系统.md) — 学术输出
