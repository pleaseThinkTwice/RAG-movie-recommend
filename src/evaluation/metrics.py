"""评估指标：Recall@K, NDCG@K, MRR。纯函数，无副作用。"""

import math
from typing import Sequence


def recall_at_k(predicted: Sequence[str], relevant: set[str], k: int = 10) -> float:
    """Recall@K：Top-K 结果中相关项占全部相关项的比例。"""
    if not relevant:
        return 0.0
    top_k = predicted[:k]
    hits = sum(1 for pid in top_k if pid in relevant)
    return hits / len(relevant)


def ndcg_at_k(predicted: Sequence[str], relevant: set[str], k: int = 10) -> float:
    """NDCG@K：归一化折扣累积增益（二元相关性）。"""
    if not relevant:
        return 0.0

    # DCG
    dcg = 0.0
    for i, pid in enumerate(predicted[:k]):
        if pid in relevant:
            dcg += 1.0 / math.log2(i + 2)  # i+2 因为 i 从 0 开始

    # IDCG (理想排序——所有 relevant 排在最前面)
    idcg = 0.0
    for i in range(min(len(relevant), k)):
        idcg += 1.0 / math.log2(i + 2)

    return dcg / idcg if idcg > 0 else 0.0


def mrr(predicted: Sequence[str], relevant: set[str]) -> float:
    """MRR：第一个 relevant 项的倒数排名。"""
    for i, pid in enumerate(predicted):
        if pid in relevant:
            return 1.0 / (i + 1)
    return 0.0
