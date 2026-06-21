"""Reciprocal Rank Fusion (RRF) — 多路检索结果融合。"""

from collections import defaultdict
from typing import Optional

from src.core.schemas import RetrievalResult


def rrf_fuse(
    result_lists: list[list[RetrievalResult]],
    k: int = 60,
    top_k: int = 50,
) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion 融合多路检索结果。

    公式: RRF(d) = Σ_i 1/(k + rank_i(d))

    Args:
        result_lists: 各路检索结果列表。每路按 rank 升序排列。
        k: RRF 常数，默认 60。
        top_k: 返回的融合后 Top-K。

    Returns:
        按 RRF 分数降序排列的 [(movie_id, rrf_score), ...]。
    """
    # 先去重到 movie 级别：同一路中同一部电影只保留最高排名
    deduped_lists: list[list[RetrievalResult]] = []
    for results in result_lists:
        seen: dict[str, RetrievalResult] = {}
        for r in results:
            if r.movie_id not in seen or r.rank < seen[r.movie_id].rank:
                seen[r.movie_id] = r
        deduped_lists.append(sorted(seen.values(), key=lambda x: x.rank))

    # RRF 融合
    scores: dict[str, float] = defaultdict(float)
    for results in deduped_lists:
        for r in results:
            scores[r.movie_id] += 1.0 / (k + r.rank)

    # 按分数降序，取 top_k
    sorted_movies = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_movies[:top_k]
