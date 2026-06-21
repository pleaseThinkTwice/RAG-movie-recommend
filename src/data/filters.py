"""电影数据过滤条件。"""


def has_valid_comment(comment: str | None, min_len: int = 20) -> bool:
    """评论字段非空且长度足够。"""
    return bool(comment) and len(str(comment).strip()) >= min_len


def has_valid_score_count(score_amt: float | int | None, min_count: int = 100) -> bool:
    """评分人数达到最低要求。"""
    if score_amt is None:
        return False
    return float(score_amt) >= min_count


def has_valid_plot(text_info: str | None, min_len: int = 30) -> bool:
    """有剧情文本且长度足够。"""
    return bool(text_info) and len(str(text_info).strip()) >= min_len


def has_genres(genres: str | None) -> bool:
    """有类型标签。"""
    return bool(genres) and len(str(genres).strip()) > 0
