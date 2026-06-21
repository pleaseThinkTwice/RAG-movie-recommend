"""评估运行器：在评估集上跑完整 pipeline，计算所有指标。"""

import json
import os
import time
from datetime import datetime
from typing import Optional

from tqdm import tqdm

from src.core.config import get_config
from src.core.schemas import EvalReport, RankedMovie
from src.evaluation.metrics import recall_at_k, ndcg_at_k, mrr
from src.evaluation.eval_set import EvalSet


class EvalRunner:
    """评估运行器。"""

    def __init__(self, pipeline_fn):
        """
        Args:
            pipeline_fn: 一个函数，接受 query: str，返回 list[RankedMovie]。
        """
        self.pipeline_fn = pipeline_fn

    def run(self, eval_set: EvalSet, version: str = "v1") -> EvalReport:
        """运行完整评估。

        Args:
            eval_set: 评估集。
            version: 版本标签（如 v1, v2, v3）。

        Returns:
            评估报告。
        """
        all_recall_10 = []
        all_recall_50 = []
        all_ndcg_10 = []
        all_mrr = []

        for labeled in tqdm(eval_set.queries, desc=f"Evaluating {version}"):
            relevant = set(labeled.relevant_movie_ids)
            if not relevant:
                continue

            # 运行 pipeline
            try:
                results = self.pipeline_fn(labeled.query)
            except Exception as e:
                print(f"  Error on query '{labeled.query[:50]}...': {e}")
                all_recall_10.append(0.0)
                all_recall_50.append(0.0)
                all_ndcg_10.append(0.0)
                all_mrr.append(0.0)
                continue

            pred_10 = [r.movie_id for r in results[:10]]
            pred_50 = [r.movie_id for r in results[:50]]

            all_recall_10.append(recall_at_k(pred_10, relevant, k=10))
            all_recall_50.append(recall_at_k(pred_50, relevant, k=50))
            all_ndcg_10.append(ndcg_at_k(pred_10, relevant, k=10))
            all_mrr.append(mrr(pred_10, relevant))

        n = len(all_recall_10)
        if n == 0:
            return EvalReport(
                version=version, recall_at_10=0, recall_at_50=0,
                ndcg_at_10=0, mrr=0, num_queries=0
            )

        return EvalReport(
            version=version,
            recall_at_10=sum(all_recall_10) / n,
            recall_at_50=sum(all_recall_50) / n,
            ndcg_at_10=sum(all_ndcg_10) / n,
            mrr=sum(all_mrr) / n,
            num_queries=n,
        )

    def save_report(self, report: EvalReport, path: str | None = None):
        """保存评估报告。"""
        if path is None:
            config = get_config()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join(
                config.evaluation.eval_results_dir,
                f"{report.version}_{timestamp}.json"
            )

        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(report.model_dump_json(indent=2, ensure_ascii=False))

        print(f"Report saved to {path}")

    @staticmethod
    def print_report(report: EvalReport):
        """打印评估报告。"""
        print(f"\n{'='*50}")
        print(f"  Evaluation Report - {report.version}")
        print(f"{'='*50}")
        print(f"  Queries evaluated: {report.num_queries}")
        print(f"  Recall@10:  {report.recall_at_10:.4f}")
        print(f"  Recall@50:  {report.recall_at_50:.4f}")
        print(f"  NDCG@10:    {report.ndcg_at_10:.4f}")
        print(f"  MRR:        {report.mrr:.4f}")
        print(f"{'='*50}")
