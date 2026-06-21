"""向量检索器。"""

from typing import Optional

import numpy as np

from src.indexing.embedder import Embedder
from src.indexing.vector_store import VectorStore
from src.core.schemas import RetrievalResult
from src.retrieval.metadata_filter import build_where_clause, FilterConstraints


class VectorRetriever:
    """基于向量相似度的检索。"""

    def __init__(self, embedder: Embedder | None = None, vector_store: VectorStore | None = None):
        self.embedder = embedder or Embedder()
        self.vector_store = vector_store or VectorStore()

    def retrieve(
        self,
        query: str,
        top_k: int = 50,
        filters: FilterConstraints | None = None,
        chunk_type: str | None = None,
    ) -> list[RetrievalResult]:
        """向量检索。

        Args:
            query: 查询文本。
            top_k: 返回数量。
            filters: 元数据过滤条件。
            chunk_type: 可选，只检索特定 chunk_type（如 'themes_style'）。

        Returns:
            按相似度降序排列的结果。
        """
        # Embed query
        query_embedding = self.embedder.encode_single(query)

        # 构建 where 子句
        where = build_where_clause(filters) if filters else None
        if chunk_type:
            chunk_filter = {"chunk_type": chunk_type}
            if where:
                where = {"$and": [where, chunk_filter]}
            else:
                where = chunk_filter

        # 查询
        results = self.vector_store.query(
            query_embedding.tolist(),
            top_k=top_k,
            where=where,
        )

        # 转换为 RetrievalResult
        retrieval_results = []
        for i in range(len(results["ids"][0])):
            chunk_id = results["ids"][0][i]
            metadata = results["metadatas"][0][i]
            distance = results["distances"][0][i]

            retrieval_results.append(RetrievalResult(
                movie_id=metadata.get("movie_id", chunk_id.split("_")[0]),
                chunk_id=chunk_id,
                score=1.0 - distance,  # ChromaDB cosine distance → similarity
                source="vector",
                rank=i + 1,
            ))

        return retrieval_results
