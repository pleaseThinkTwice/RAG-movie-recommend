"""元数据过滤条件构建测试。"""

from src.retrieval.metadata_filter import build_where_clause, has_filters
from src.core.schemas import FilterConstraints


def test_empty_filters():
    assert build_where_clause(FilterConstraints()) is None
    assert not has_filters(FilterConstraints())


def test_year_range():
    filters = FilterConstraints(year_min=2010, year_max=2020)
    clause = build_where_clause(filters)
    assert clause is not None
    assert "$and" in clause
    conditions = clause["$and"]
    assert {"year": {"$gte": 2010}} in conditions
    assert {"year": {"$lte": 2020}} in conditions


def test_single_genre():
    filters = FilterConstraints(genres=["科幻"])
    clause = build_where_clause(filters)
    assert clause == {"genres": {"$contains": "科幻"}}


def test_multiple_genres():
    filters = FilterConstraints(genres=["科幻", "动作"])
    clause = build_where_clause(filters)
    assert "$or" in clause or "$and" in clause


def test_rating_min():
    filters = FilterConstraints(rating_min=7.5)
    clause = build_where_clause(filters)
    assert clause == {"rating_douban": {"$gte": 7.5}}


def test_has_filters_true():
    assert has_filters(FilterConstraints(year_min=2010))
    assert has_filters(FilterConstraints(genres=["科幻"]))
    assert has_filters(FilterConstraints(rating_min=8.0))


def test_has_filters_false():
    assert not has_filters(FilterConstraints())
