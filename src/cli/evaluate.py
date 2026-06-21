"""CLI: 运行评估。"""

import sys
import io
import os

from src.core.config import get_config
from src.evaluation.eval_set import EvalSet
from src.evaluation.runner import EvalRunner


def main():
    if sys.stdout.encoding != 'utf-8':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    config = get_config()

    # 加载或创建评估集
    eval_path = config.evaluation.eval_set_path
    if not os.path.exists(eval_path):
        print(f"Creating seed eval set at {eval_path}...")
        eval_set = EvalSet.create_seed_set()
        eval_set.save(eval_path)
        print(f"  Created {len(eval_set)} queries (labels empty — needs manual annotation)")
    else:
        eval_set = EvalSet.load(eval_path)
        labeled_count = sum(1 for q in eval_set.queries if q.relevant_movie_ids)
        print(f"Loaded eval set: {len(eval_set)} queries, {labeled_count} labeled")

    # 简化版 pipeline（不用 reranker 和 explainer，纯召回+精排）
    def quick_pipeline(query: str):
        from src.query.parser import QueryParser
        from src.retrieval.orchestrator import RecallOrchestrator
        from src.ranking.reranker import Reranker
        from src.core.schemas import Chunk, RankedMovie

        parser = QueryParser()
        parsed = parser.parse_sync(query)

        orchestrator = RecallOrchestrator()
        recall_results = orchestrator.recall(parsed)

        # 简单聚合（不跑 reranker，直接用 RRF 分数）
        results: list[RankedMovie] = []
        seen: set[str] = set()
        for r in sorted(recall_results, key=lambda x: x.score, reverse=True):
            if r.movie_id in seen:
                continue
            seen.add(r.movie_id)
            results.append(RankedMovie(
                movie_id=r.movie_id, score=r.score,
                title_zh="", year=None, genres=[], director=[],
                rating_douban=None,
            ))
            if len(results) >= 50:
                break

        return results

    # 运行评估
    runner = EvalRunner(quick_pipeline)

    # v1 baseline: 纯 RRF 融合（无 reranker）
    report_v1 = runner.run(eval_set, version="v1_rrf_only")
    runner.print_report(report_v1)
    runner.save_report(report_v1)

    print("\nNote: This is a quick evaluation with RRF-fusion only (no reranker).")
    print("To get full numbers, annotate relevant_movie_ids in the eval set first.")


if __name__ == "__main__":
    main()
