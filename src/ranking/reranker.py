"""Cross-Encoder 精排：bge-reranker-large 重排序 Top-50 → Top-10。"""

from typing import Optional

import torch
from sentence_transformers import CrossEncoder

from src.core.config import get_config, RerankerConfig
from src.core.schemas import Chunk, RetrievalResult


class Reranker:
    """Cross-Encoder 精排器。"""

    def __init__(self, config: Optional[RerankerConfig] = None):
        if config is None:
            config = get_config().reranker
        self.config = config

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading reranker model: {config.model_name} on {device}...")
        self.model = CrossEncoder(config.model_name, device=device)
        self.model.model.eval()

    def _truncate_chunk(self, chunk: Chunk) -> str:
        """截断长 chunk，保留关键信息。"""
        text = chunk.text
        if len(text) <= self.config.truncate_chars:
            return text

        # 构造: 标题 | 类型 | 前 N 字
        prefix = f"{chunk.title_zh}"
        if chunk.genres:
            prefix += f" | {'/'.join(chunk.genres)}"
        truncated = text[:self.config.truncate_chars]
        return f"{prefix} | {truncated}"

    def rerank(
        self,
        query: str,
        chunks: list[Chunk],
        top_k: int | None = None,
    ) -> list[tuple[Chunk, float]]:
        """对候选 chunk 批量精排。

        Args:
            query: 查询文本。
            chunks: 候选 chunk 列表（每个代表一个 movie）。
            top_k: 返回 Top-K。

        Returns:
            按相关性分数降序排列的 [(chunk, score), ...]。
        """
        if top_k is None:
            top_k = get_config().rerank.top_k

        if not chunks:
            return []

        # 构建 (query, chunk_text) 对
        pairs = [(query, self._truncate_chunk(c)) for c in chunks]

        # 批量推理
        scores = self.model.predict(
            pairs,
            batch_size=self.config.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        # 排序
        scored = list(zip(chunks, scores))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def rerank_from_results(
        self,
        query: str,
        recall_results: list[RetrievalResult],
        all_chunks: dict[str, Chunk],
        top_k: int | None = None,
    ) -> list[tuple[Chunk, float]]:
        """从召回结果中选取代表 chunk 并精排。

        每个 movie 选最高分的 chunk 作为代表。
        """
        # 去重到 movie 级别，保留每部电影最高分的 chunk
        movie_best_chunk: dict[str, tuple[Chunk, float]] = {}
        for r in recall_results:
            if r.chunk_id not in all_chunks:
                continue
            chunk = all_chunks[r.chunk_id]
            if (r.movie_id not in movie_best_chunk or
                    r.score > movie_best_chunk[r.movie_id][1]):
                movie_best_chunk[r.movie_id] = (chunk, r.score)

        chunks = [c for c, _ in movie_best_chunk.values()]
        return self.rerank(query, chunks, top_k)
