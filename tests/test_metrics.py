"""评估指标单元测试。"""

from src.evaluation.metrics import recall_at_k, ndcg_at_k, mrr


def test_recall_perfect():
    assert recall_at_k(["A", "B", "C"], {"A", "B"}, k=3) == 1.0


def test_recall_partial():
    assert recall_at_k(["A", "B", "C"], {"A", "D", "E"}, k=3) == 1/3


def test_recall_empty_relevant():
    assert recall_at_k(["A", "B"], set(), k=10) == 0.0


def test_recall_k_smaller_than_results():
    assert recall_at_k(["A", "B", "C", "D"], {"A", "D"}, k=2) == 0.5


def test_ndcg_perfect():
    # 前 3 个全是 relevant → DCG = IDCG
    assert ndcg_at_k(["A", "B", "C"], {"A", "B", "C"}, k=3) == 1.0


def test_ndcg_suboptimal():
    # relevant 在第二位
    score = ndcg_at_k(["X", "A", "Y"], {"A"}, k=3)
    assert 0 < score < 1.0


def test_ndcg_empty():
    assert ndcg_at_k(["A", "B"], set(), k=10) == 0.0


def test_mrr_first():
    assert mrr(["A", "B", "C"], {"A"}) == 1.0


def test_mrr_third():
    assert mrr(["X", "Y", "A"], {"A"}) == 1.0 / 3


def test_mrr_none():
    assert mrr(["X", "Y", "Z"], {"A"}) == 0.0
