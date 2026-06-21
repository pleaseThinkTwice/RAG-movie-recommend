"""元数据过滤：将 FilterConstraints 翻译为 ChromaDB where 子句。"""

from typing import Optional

from src.core.schemas import FilterConstraints


def build_where_clause(filters: FilterConstraints) -> Optional[dict]:
    """构建 ChromaDB where 过滤条件。

    Args:
        filters: 解析出的过滤约束。

    Returns:
        ChromaDB where dict，无约束时返回 None。
    """
    conditions = []

    # 年份范围
    if filters.year_min is not None:
        conditions.append({"year": {"$gte": filters.year_min}})
    if filters.year_max is not None:
        conditions.append({"year": {"$lte": filters.year_max}})

    # 评分下限
    if filters.rating_min is not None:
        conditions.append({"rating_douban": {"$gte": filters.rating_min}})

    # 类型（ChromaDB $contains 只能处理单个值，多类型用 $or）
    if filters.genres and len(filters.genres) > 0:
        if len(filters.genres) == 1:
            conditions.append({"genres": {"$contains": filters.genres[0]}})
        else:
            genre_conditions = [
                {"genres": {"$contains": g}} for g in filters.genres
            ]
            conditions.append({"$or": genre_conditions})

    # 导演（ChromaDB $contains 在 list 字段上同样适用）
    if filters.directors and len(filters.directors) > 0:
        if len(filters.directors) == 1:
            conditions.append({"director": {"$contains": filters.directors[0]}})
        else:
            dir_conditions = [
                {"director": {"$contains": d}} for d in filters.directors
            ]
            conditions.append({"$or": dir_conditions})

    # 国家/地区
    if filters.countries and len(filters.countries) > 0:
        for country in filters.countries:
            conditions.append({"country": {"$contains": country}})

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


def has_filters(filters: FilterConstraints) -> bool:
    """检查是否有非空过滤条件。"""
    return any([
        filters.year_min is not None,
        filters.year_max is not None,
        filters.rating_min is not None,
        bool(filters.genres),
        bool(filters.directors),
        bool(filters.countries),
    ])
