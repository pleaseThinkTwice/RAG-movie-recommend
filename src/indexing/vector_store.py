"""ChromaDB 向量库封装。"""

import json
import os
from typing import Optional

import chromadb
from chromadb.config import Settings as ChromaSettings
from tqdm import tqdm

from src.core.config import get_config, ChromaDBConfig
from src.core.schemas import Chunk


class VectorStore:
    """ChromaDB 向量存储封装。"""

    def __init__(self, config: Optional[ChromaDBConfig] = None):
        if config is None:
            config = get_config().chromadb
        self.config = config

        os.makedirs(config.persist_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=config.persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )

    @property
    def collection(self):
        """获取或创建 collection。"""
        return self.client.get_or_create_collection(
            name=self.config.collection_name,
            metadata={"hnsw:space": self.config.hnsw_space},
        )

    def rebuild_collection(self):
        """删除并重建 collection。"""
        try:
            self.client.delete_collection(self.config.collection_name)
        except Exception:
            pass
        return self.collection

    def add_chunks(
        self,
        chunks: list[Chunk],
        embeddings: list[list[float]],
        batch_size: int = 5000,
    ):
        """批量添加 chunk 和嵌入向量到 ChromaDB。"""
        collection = self.collection
        for i in tqdm(range(0, len(chunks), batch_size), desc="  Adding to ChromaDB"):
            batch_chunks = chunks[i:i + batch_size]
            batch_embeddings = embeddings[i:i + batch_size]

            ids = [c.chunk_id for c in batch_chunks]
            docs = [c.text for c in batch_chunks]
            metas = [
                {
                    "movie_id": c.movie_id,
                    "chunk_type": c.chunk_type,
                    "title_zh": c.title_zh,
                    "year": c.year or 0,
                    "genres": c.genres,       # ChromaDB 支持 list 类型
                    "director": c.director,
                    "rating_douban": c.rating_douban or 0.0,
                }
                for c in batch_chunks
            ]

            collection.add(
                ids=ids,
                documents=docs,
                embeddings=batch_embeddings,
                metadatas=metas,
            )

    def query(
        self,
        query_embedding: list[float],
        top_k: int = 50,
        where: dict | None = None,
        where_document: dict | None = None,
    ) -> dict:
        """检索最相似的 chunk。

        Returns:
            ChromaDB 查询结果字典，包含 ids, distances, metadatas, documents。
        """
        collection = self.collection
        return collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            where_document=where_document,
            include=["documents", "metadatas", "distances"],
        )

    def count(self) -> int:
        """返回 collection 中的文档数。"""
        return self.collection.count()
