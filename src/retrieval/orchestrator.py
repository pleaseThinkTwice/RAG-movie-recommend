"""召回编排器：三路并行检索 + RRF 融合。"""

import asyncio
from typing import Optional

from src.core.config import get_config
from src.core.schemas import ParsedQuery, RetrievalResult
from src.retrieval.rrf import rrf_fuse
from src.retrieval.metadata_filter import has_filters
from src.retrieval.vector_retriever import VectorRetriever
from src.retrieval.bm25_retriever import BM25Retriever
from src.indexing.embedder import Embedder
from src.indexing.vector_store import VectorStore
from src.indexing.bm25_index import BM25Searcher


class RecallOrchestrator:
    """召回编排器：协调多路检索并 RRF 融合。"""

    def __init__(
        self,
        embedder: Embedder | None = None,
        vector_store: VectorStore | None = None,
        bm25: BM25Searcher | None = None,
    ):
        self.embedder = embedder or Embedder()
        self.vector_store = vector_store or VectorStore()
        self.vector_retriever = VectorRetriever(self.embedder, self.vector_store)
        self.bm25_retriever = BM25Retriever(bm25 or BM25Searcher())
        self.config = get_config().recall

    def recall(self, parsed_query: ParsedQuery) -> list[RetrievalResult]:
        """同步召回：三路并行 + RRF 融合 → Top-50。

        Args:
            parsed_query: 解析后的结构化查询。

        Returns:
            融合后的 Top-K 检索结果。
        """
        # 如果 BM25 需要元数据后置过滤，先跑向量检索获取候选集
        has_f = has_filters(parsed_query.filters)
        allowed_ids: set[str] | None = None

        if has_f:
            # 用向量检索先获取满足元数据过滤条件的候选 movie_ids
            vec_with_filters = self.vector_retriever.retrieve(
                parsed_query.semantic_query,
                top_k=200,
                filters=parsed_query.filters,
            )
            allowed_ids = {r.movie_id for r in vec_with_filters}

        # 三路召回
        results_lists: list[list[RetrievalResult]] = []

        # 路 1: 向量检索（无过滤，让 RRF 自己排序）
        vec_results = self.vector_retriever.retrieve(
            parsed_query.semantic_query,
            top_k=self.config.vector_top_k,
        )
        results_lists.append(vec_results)

        # 路 2: BM25 检索
        bm25_results = self.bm25_retriever.retrieve(
            parsed_query.semantic_query,
            top_k=self.config.bm25_top_k,
            allowed_movie_ids=allowed_ids,
        )
        results_lists.append(bm25_results)

        # 路 3: similar_to（如果有引用电影）
        if parsed_query.similar_to:
            from src.query.similar_to import SimilarToResolver
            resolver = SimilarToResolver.from_movies_jsonl()
            for ref_name in parsed_query.similar_to:
                ref_id = resolver.resolve(ref_name)
                if ref_id:
                    # 获取该电影 themes_style chunk 的 embedding，搜相似电影
                    # 简化：直接用电影名作为额外查询
                    similar_results = self.vector_retriever.retrieve(
                        ref_name,
                        top_k=self.config.vector_top_k,
                        chunk_type="themes_style",
                    )
                    # 标记来源
                    for r in similar_results:
                        r.source = "similar_to"
                    results_lists.append(similar_results)

        # RRF 融合
        fused_tuples = rrf_fuse(
            results_lists,
            k=self.config.rrf_k,
            top_k=self.config.fusion_top_k,
        )

        # 兜底：结果太少时放宽
        if len(fused_tuples) < self.config.min_results_fallback:
            extra = self.vector_retriever.retrieve(
                parsed_query.semantic_query,
                top_k=50,
            )
            extra_list = [[r for r in extra if r.movie_id not in dict(fused_tuples)]]
            fused_tuples = rrf_fuse(
                [results_lists[0], results_lists[1]] + extra_list,
                k=self.config.rrf_k,
                top_k=self.config.fusion_top_k,
            )

        # 转换回 RetrievalResult
        fused_results = []
        for movie_id, rrf_score in fused_tuples:
            fused_results.append(RetrievalResult(
                movie_id=movie_id,
                chunk_id="",
                score=rrf_score,
                source="rrf_fusion",
                rank=len(fused_results) + 1,
            ))
        return fused_results

    async def recall_async(self, parsed_query: ParsedQuery) -> list[RetrievalResult]:
        """异步版本（后续可优化为真正的并行检索）。"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.recall, parsed_query)
