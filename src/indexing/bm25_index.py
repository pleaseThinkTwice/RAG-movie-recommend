"""BM25 关键词检索索引。"""

import json
import os
import pickle
from typing import Optional

import jieba
from rank_bm25 import BM25Okapi
from tqdm import tqdm

from src.core.config import get_config, BM25Config
from src.core.schemas import Chunk, RetrievalResult


class BM25Searcher:
    """BM25 检索器，基于 jieba 分词。"""

    def __init__(self, config: Optional[BM25Config] = None):
        if config is None:
            config = get_config().bm25
        self.config = config
        self.index: BM25Okapi | None = None
        self.chunks: list[Chunk] = []

    def build(self, chunks: list[Chunk]):
        """构建 BM25 索引。"""
        self.chunks = chunks
        print(f"  Tokenizing {len(chunks)} chunks with jieba...")
        tokenized = []
        for c in tqdm(chunks, desc="  Tokenizing"):
            tokens = list(jieba.cut(c.text))
            tokenized.append(tokens)

        print(f"  Building BM25Okapi index (k1={self.config.k1}, b={self.config.b})...")
        self.index = BM25Okapi(tokenized, k1=self.config.k1, b=self.config.b)
        print(f"  BM25 index built: {len(tokenized)} documents")

    def search(
        self,
        query: str,
        top_k: int = 50,
        allowed_movie_ids: set[str] | None = None,
    ) -> list[RetrievalResult]:
        """搜索并返回 top-k 结果。

        Args:
            query: 查询文本。
            top_k: 返回数量。
            allowed_movie_ids: 如果提供，只返回这些 movie_id 的结果（元数据后置过滤）。
                此时 BM25 内部先取 top_k * 2，再过滤。

        Returns:
            按 BM25 分数降序排列的结果。
        """
        if self.index is None:
            raise RuntimeError("BM25 index not built. Call build() first.")

        query_tokens = list(jieba.cut(query))
        scores = self.index.get_scores(query_tokens)

        # 排序索引
        fetch_k = top_k * 2 if allowed_movie_ids else top_k
        fetch_k = min(fetch_k, len(scores))

        # 获取 top fetch_k 的索引
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        if allowed_movie_ids:
            top_indices = [i for i in top_indices if self.chunks[i].movie_id in allowed_movie_ids]

        results = []
        for rank, idx in enumerate(top_indices[:top_k]):
            chunk = self.chunks[idx]
            results.append(RetrievalResult(
                movie_id=chunk.movie_id,
                chunk_id=chunk.chunk_id,
                score=float(scores[idx]),
                source="bm25",
                rank=rank + 1,
            ))
        return results

    def save(self, path: str | None = None):
        """保存索引到磁盘（含 tokenized 数据，加载秒级）。"""
        if path is None:
            path = self.config.index_path
        if self.index is None:
            raise RuntimeError("No index to save. Call build() first.")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = {
            "chunks": [c.model_dump() for c in self.chunks],
            "index": self.index,  # BM25Okapi 可直接 pickle
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        print(f"BM25 index saved to {path}")

    def load(self, path: str | None = None):
        """从磁盘加载索引（直接加载 tokenized 数据，无需重建）。"""
        if path is None:
            path = self.config.index_path
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.chunks = [Chunk(**c) for c in data["chunks"]]
        self.index = data["index"]
        print(f"BM25 index loaded from {path} ({len(self.chunks)} docs)")
        return self
