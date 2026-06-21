"""RRF 融合单元测试。"""

from src.retrieval.rrf import rrf_fuse
from src.core.schemas import RetrievalResult


def test_rrf_basic():
    """基础 RRF 融合：两路检索，不同排序。"""
    list_a = [
        RetrievalResult(movie_id="A", chunk_id="A_1", score=0.9, source="vector", rank=1),
        RetrievalResult(movie_id="B", chunk_id="B_1", score=0.8, source="vector", rank=2),
        RetrievalResult(movie_id="C", chunk_id="C_1", score=0.7, source="vector", rank=3),
    ]
    list_b = [
        RetrievalResult(movie_id="B", chunk_id="B_2", score=5.0, source="bm25", rank=1),
        RetrievalResult(movie_id="C", chunk_id="C_2", score=4.0, source="bm25", rank=2),
        RetrievalResult(movie_id="A", chunk_id="A_2", score=3.0, source="bm25", rank=3),
    ]

    fused = rrf_fuse([list_a, list_b], k=60, top_k=5)

    assert len(fused) == 3
    # B 在两个列表中都排前面，应该总分最高
    assert fused[0][0] == "B"
    assert fused[0][1] > fused[1][1] > fused[2][1]


def test_rrf_missing_from_one_list():
    """某部电影只在一路结果中出现。"""
    list_a = [
        RetrievalResult(movie_id="A", chunk_id="A_1", score=0.9, source="vector", rank=1),
        RetrievalResult(movie_id="B", chunk_id="B_1", score=0.8, source="vector", rank=2),
    ]
    list_b = [
        RetrievalResult(movie_id="C", chunk_id="C_1", score=5.0, source="bm25", rank=1),
    ]

    fused = rrf_fuse([list_a, list_b], k=60, top_k=5)
    ids = [f[0] for f in fused]
    assert "A" in ids
    assert "B" in ids
    assert "C" in ids


def test_rrf_dedupe_same_movie():
    """一路中同一部电影有多个 chunk，应只保留最高排名。"""
    list_a = [
        RetrievalResult(movie_id="A", chunk_id="A_plot", score=0.9, source="vector", rank=1),
        RetrievalResult(movie_id="A", chunk_id="A_themes", score=0.85, source="vector", rank=2),
        RetrievalResult(movie_id="B", chunk_id="B_plot", score=0.8, source="vector", rank=3),
    ]
    list_b = [
        RetrievalResult(movie_id="B", chunk_id="B_reviews", score=5.0, source="bm25", rank=1),
    ]

    fused = rrf_fuse([list_a, list_b], k=60, top_k=5)
    # A 应该只出现一次，且 rank 取 1
    assert len(fused) == 2
    assert fused[0][0] == "B"  # B gets points from both lists (rank 3 + rank 1) = 1/63+1/61
    # A gets 1/61 (rank 1 in list_a)
    # Actually: A = 1/61 ≈ 0.01639, B = 1/63 + 1/61 ≈ 0.01587 + 0.01639 = 0.03226
    # So B > A indeed


def test_rrf_empty_input():
    """空输入不报错。"""
    fused = rrf_fuse([], k=60, top_k=50)
    assert fused == []


def test_rrf_top_k_truncation():
    """验证 top_k 截断。"""
    results = []
    for i in range(100):
        results.append(
            RetrievalResult(movie_id=f"M{i}", chunk_id=f"M{i}_c", score=1.0, source="vector", rank=i+1)
        )
    fused = rrf_fuse([results], k=60, top_k=10)
    assert len(fused) == 10
