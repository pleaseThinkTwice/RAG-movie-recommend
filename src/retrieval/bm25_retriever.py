"""BM25 检索器封装。"""

from typing import Optional

from src.indexing.bm25_index import BM25Searcher
from src.core.schemas import RetrievalResult, FilterConstraints
from src.retrieval.metadata_filter import has_filters


class BM25Retriever:
    """BM25 关键词检索封装。"""

    def __init__(self, bm25: BM25Searcher | None = None):
        self.bm25 = bm25 or BM25Searcher()
        if self.bm25.index is None:
            self.bm25.load()

    def retrieve(
        self,
        query: str,
        top_k: int = 50,
        allowed_movie_ids: set[str] | None = None,
    ) -> list[RetrievalResult]:
        """BM25 检索。

        Args:
            query: 查询文本。
            top_k: 返回数量。
            allowed_movie_ids: 可选的 movie_id 白名单（元数据后置过滤）。

        Returns:
            按 BM25 分数降序排列的结果。
        """
        return self.bm25.search(
            query=query,
            top_k=top_k,
            allowed_movie_ids=allowed_movie_ids,
        )
