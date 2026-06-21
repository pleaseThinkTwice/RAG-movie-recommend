"""评估集加载与管理。"""

import json
import os
from pathlib import Path
from typing import Optional

from src.core.schemas import LabeledQuery


class EvalSet:
    """标注评估集。"""

    def __init__(self, queries: list[LabeledQuery]):
        self.queries = queries

    @classmethod
    def load(cls, path: str | None = None) -> "EvalSet":
        """从 JSONL 文件加载评估集。"""
        if path is None:
            from src.core.config import get_config
            path = get_config().evaluation.eval_set_path

        if not os.path.exists(path):
            print(f"Eval set not found at {path}. Creating empty set.")
            return cls([])

        queries = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    queries.append(LabeledQuery.model_validate_json(line))
        return cls(queries)

    def save(self, path: str | None = None):
        """保存评估集到 JSONL 文件。"""
        if path is None:
            from src.core.config import get_config
            path = get_config().evaluation.eval_set_path

        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for q in self.queries:
                f.write(q.model_dump_json(ensure_ascii=False) + "\n")

    @classmethod
    def create_seed_set(cls) -> "EvalSet":
        """创建种子评估集（~20 条手工 query，覆盖四种类型）。"""
        seed = [
            # --- Mood queries ---
            LabeledQuery(
                query="想看烧脑的科幻片，不要太晦涩，节奏快一点",
                relevant_movie_ids=[], source="manual"
            ),
            LabeledQuery(
                query="治愈系电影，看完会觉得很温暖那种",
                relevant_movie_ids=[], source="manual"
            ),
            LabeledQuery(
                query="让人看完压抑到说不出话的电影",
                relevant_movie_ids=[], source="manual"
            ),
            LabeledQuery(
                query="热血励志，看完想立刻去努力的那种",
                relevant_movie_ids=[], source="manual"
            ),
            LabeledQuery(
                query="结局反转让人目瞪口呆的悬疑片",
                relevant_movie_ids=[], source="manual"
            ),
            # --- Entity queries ---
            LabeledQuery(
                query="诺兰导演的电影",
                relevant_movie_ids=[], source="manual"
            ),
            LabeledQuery(
                query="莱昂纳多主演的经典电影",
                relevant_movie_ids=[], source="manual"
            ),
            LabeledQuery(
                query="宫崎骏的动画电影",
                relevant_movie_ids=[], source="manual"
            ),
            # --- Hybrid queries ---
            LabeledQuery(
                query="2010年以后的烧脑科幻片，评分8分以上",
                relevant_movie_ids=[], source="manual"
            ),
            LabeledQuery(
                query="2000年以前的经典爱情片",
                relevant_movie_ids=[], source="manual"
            ),
            LabeledQuery(
                query="中国的青春片，评分7.5以上",
                relevant_movie_ids=[], source="manual"
            ),
            # --- Similar-to queries ---
            LabeledQuery(
                query="像《盗梦空间》那样烧脑的科幻电影",
                relevant_movie_ids=[], source="manual"
            ),
            LabeledQuery(
                query="类似《你的名字》的日本动画电影",
                relevant_movie_ids=[], source="manual"
            ),
            LabeledQuery(
                query="和《这个杀手不太冷》一样感人的电影",
                relevant_movie_ids=[], source="manual"
            ),
            # --- Douban list adapted ---
            LabeledQuery(
                query="硬核科幻电影推荐",
                relevant_movie_ids=[], source="douban_list"
            ),
            LabeledQuery(
                query="那些看完让你思考人生的电影",
                relevant_movie_ids=[], source="douban_list"
            ),
            LabeledQuery(
                query="适合一个人安静看的文艺片",
                relevant_movie_ids=[], source="douban_list"
            ),
            # --- LLM generated ---
            LabeledQuery(
                query="慢节奏但很深刻的历史题材电影",
                relevant_movie_ids=[], source="llm_generated"
            ),
            LabeledQuery(
                query="赛博朋克风格的黑暗科幻片",
                relevant_movie_ids=[], source="llm_generated"
            ),
            LabeledQuery(
                query="真实事件改编的战争片，催泪的那种",
                relevant_movie_ids=[], source="llm_generated"
            ),
        ]
        return cls(seed)

    def __len__(self) -> int:
        return len(self.queries)

    def __getitem__(self, idx) -> LabeledQuery:
        return self.queries[idx]
