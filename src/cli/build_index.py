"""CLI: 构建全部索引（Embedding + ChromaDB + BM25）。"""

import json
import os
import sys
import io

import numpy as np

from src.core.config import get_config
from src.core.schemas import Chunk
from src.indexing.embedder import Embedder
from src.indexing.vector_store import VectorStore
from src.indexing.bm25_index import BM25Searcher


def main():
    # 确保 UTF-8 输出
    if sys.stdout.encoding != 'utf-8':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    config = get_config()

    # 1. 加载 chunks
    chunks_path = config.data.chunks_file
    print(f"Loading chunks from {chunks_path}...")
    chunks: list[Chunk] = []
    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(Chunk.model_validate_json(line.strip()))
    print(f"  Loaded {len(chunks)} chunks")

    # 2. 初始化 Embedder
    embedder = Embedder()

    # 3. 编码所有 chunks（分批，节省内存）
    print(f"\nEncoding {len(chunks)} chunks...")
    batch_size = 4096  # 大 batch 加速 GPU 编码
    all_embeddings: list[np.ndarray] = []

    for i in range(0, len(chunks), batch_size):
        batch_texts = [c.text for c in chunks[i:i + batch_size]]
        embs = embedder.encode_batch(batch_texts, batch_size=config.embedding.batch_size, show_progress=False)
        all_embeddings.append(embs)
        if (i // batch_size) % 5 == 0:
            print(f"  Encoded {min(i + batch_size, len(chunks))}/{len(chunks)}")

    all_embeddings = np.concatenate(all_embeddings, axis=0)
    print(f"  Embeddings shape: {all_embeddings.shape}")

    # 4. 写入 ChromaDB
    print(f"\nBuilding ChromaDB index...")
    vector_store = VectorStore()
    vector_store.rebuild_collection()
    vector_store.add_chunks(chunks, all_embeddings.tolist())
    print(f"  ChromaDB collection size: {vector_store.count()}")

    # 5. 构建 BM25 索引
    print(f"\nBuilding BM25 index...")
    bm25 = BM25Searcher()
    bm25.build(chunks)
    bm25.save()
    print(f"  BM25 index built and saved")

    # 6. 快速验证
    print(f"\n=== Verification ===")
    test_query = "烧脑科幻大片"
    print(f"Test query: '{test_query}'")

    # 向量检索
    q_emb = embedder.encode_single(test_query)
    vec_results = vector_store.query(q_emb.tolist(), top_k=5)
    print(f"\nVector retrieval top-5:")
    for i, (mid, dist) in enumerate(zip(
        vec_results["metadatas"][0],
        vec_results["distances"][0]
    )):
        print(f"  {i+1}. {mid['title_zh']} ({mid['year']}) - dist={dist:.4f} - [{mid['chunk_type']}]")

    # BM25 检索
    bm25_results = bm25.search(test_query, top_k=5)
    print(f"\nBM25 retrieval top-5:")
    for r in bm25_results:
        chunk = [c for c in chunks if c.chunk_id == r.chunk_id][0]
        print(f"  {r.rank}. {chunk.title_zh} ({chunk.year}) - score={r.score:.4f} - [{chunk.chunk_type}]")

    print(f"\nIndex build complete!")


if __name__ == "__main__":
    main()
