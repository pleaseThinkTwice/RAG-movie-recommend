"""配置加载模块。读取 config/default.yaml，支持环境变量插值。"""

import os
from pathlib import Path
from dataclasses import dataclass, field

import yaml


@dataclass
class EmbeddingConfig:
    model_name: str = "BAAI/bge-large-zh-v1.5"
    dimension: int = 1024
    max_tokens: int = 512
    batch_size: int = 32
    normalize: bool = True


@dataclass
class RerankerConfig:
    model_name: str = "BAAI/bge-reranker-large"
    max_tokens: int = 512
    truncate_chars: int = 400
    batch_size: int = 16


@dataclass
class LLMConfig:
    api_key: str = ""
    base_url: str = "https://api.deepseek.com/v1"
    model: str = "deepseek-chat"
    temperature_query_parse: float = 0.0
    temperature_generation: float = 0.3
    max_tokens_generation: int = 256


@dataclass
class ChromaDBConfig:
    persist_dir: str = "data/indexes/chroma"
    collection_name: str = "movies"
    hnsw_space: str = "cosine"


@dataclass
class BM25Config:
    k1: float = 1.5
    b: float = 0.75
    index_path: str = "data/indexes/bm25.pkl"


@dataclass
class RecallConfig:
    vector_top_k: int = 50
    bm25_top_k: int = 50
    rrf_k: int = 60
    fusion_top_k: int = 50
    min_results_fallback: int = 10


@dataclass
class RerankConfig:
    top_k: int = 10


@dataclass
class DataFilterConfig:
    min_comment_len: int = 20
    min_score_count: int = 100
    min_text_info_len: int = 30


@dataclass
class DataConfig:
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    movies_file: str = "data/processed/movies.jsonl"
    chunks_file: str = "data/processed/chunks.jsonl"
    themes_cache: str = "data/processed/themes_cache.jsonl"
    style_dict: str = "config/style_dict.json"
    filter: DataFilterConfig = field(default_factory=DataFilterConfig)


@dataclass
class EvaluationConfig:
    eval_set_path: str = "data/eval/eval_set.jsonl"
    eval_results_dir: str = "data/eval_results"
    judge_sample_size: int = 30


@dataclass
class Config:
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    reranker: RerankerConfig = field(default_factory=RerankerConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    chromadb: ChromaDBConfig = field(default_factory=ChromaDBConfig)
    bm25: BM25Config = field(default_factory=BM25Config)
    recall: RecallConfig = field(default_factory=RecallConfig)
    rerank: RerankConfig = field(default_factory=RerankConfig)
    data: DataConfig = field(default_factory=DataConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)


def _load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _deep_update(base: dict, override: dict) -> dict:
    for key, value in override.items():
        if isinstance(value, dict) and key in base and isinstance(base[key], dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _dataclass_from_dict(cls, data: dict | None) -> object:
    """递归地将嵌套字典转换为嵌套 dataclass。"""
    if data is None:
        return cls()
    import dataclasses
    field_types = {f.name: f.type for f in dataclasses.fields(cls)}
    kwargs = {}
    for key, value in data.items():
        if key not in field_types:
            continue  # YAML 中有但 dataclass 里没有的字段，跳过（如 api_key_env）
        elif dataclasses.is_dataclass(field_types[key]) and isinstance(value, dict):
            kwargs[key] = _dataclass_from_dict(field_types[key], value)
        else:
            kwargs[key] = value
    return cls(**kwargs)


def load_config(config_path: str | None = None) -> Config:
    """加载配置：从 default.yaml 读取，环境变量覆盖 API key。"""
    if config_path is None:
        config_path = Path(__file__).parent.parent.parent / "config" / "default.yaml"

    raw = _load_yaml(Path(config_path))

    # 环境变量覆盖 LLM API key
    llm_cfg = raw.get("llm", {})
    api_key_env = llm_cfg.get("api_key_env", "DEEPSEEK_API_KEY")
    llm_cfg["api_key"] = os.environ.get(api_key_env, "")

    return _dataclass_from_dict(Config, raw)


# 全局单例
_config: Config | None = None


def get_config() -> Config:
    global _config
    if _config is None:
        _config = load_config()
    return _config
