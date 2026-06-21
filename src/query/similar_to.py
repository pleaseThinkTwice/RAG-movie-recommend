"""Similar-to 解析器：将用户引用的电影名解析为 movie_id。"""

from typing import Optional


class SimilarToResolver:
    """根据电影名查找 movie_id。"""

    def __init__(self, title_to_id: dict[str, str] | None = None):
        """
        Args:
            title_to_id: {电影名: movie_id} 的映射表。
        """
        self.title_to_id: dict[str, str] = title_to_id or {}
        self._lower_map: dict[str, str] = {k.lower(): v for k, v in self.title_to_id.items()}

    @classmethod
    def from_movies_jsonl(cls, movies_path: str = "data/processed/movies.jsonl") -> "SimilarToResolver":
        """从 movies.jsonl 加载标题→ID 映射。"""
        import json
        mapping = {}
        with open(movies_path, "r", encoding="utf-8") as f:
            for line in f:
                m = json.loads(line)
                name = m.get("title_zh", "")
                if name:
                    mapping[name] = m["movie_id"]
        return cls(mapping)

    def resolve(self, movie_name: str) -> Optional[str]:
        """查找电影名对应的 movie_id。

        策略：
        1. 精确匹配
        2. 忽略大小写匹配
        3. 包含匹配（用户输入是电影名的子串）
        """
        if not movie_name:
            return None

        # 1. 精确匹配
        if movie_name in self.title_to_id:
            return self.title_to_id[movie_name]

        # 2. 忽略大小写
        lower_name = movie_name.lower()
        if lower_name in self._lower_map:
            return self._lower_map[lower_name]

        # 3. 包含匹配——用户输入是完整电影名的子串
        for title, mid in self.title_to_id.items():
            if lower_name in title.lower() or title.lower() in lower_name:
                return mid

        return None

    def resolve_batch(self, movie_names: list[str]) -> list[tuple[str, Optional[str]]]:
        """批量解析。返回 [(原名, movie_id or None), ...]"""
        return [(name, self.resolve(name)) for name in movie_names]
