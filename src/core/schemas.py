"""数据模型定义。"""

from __future__ import annotations

import json
from typing import Optional

from pydantic import BaseModel, Field


class Movie(BaseModel):
    """电影完整信息。"""
    movie_id: str                                    # 主键（优先 IMDB ID，否则豆瓣 ID）
    title_zh: str = ""
    title_en: str = ""
    year: int | None = None
    genres: list[str] = Field(default_factory=list)
    director: list[str] = Field(default_factory=list)
    cast: list[str] = Field(default_factory=list)
    rating_douban: float | None = None
    rating_count: int | None = None
    runtime: int | None = None
    country: list[str] = Field(default_factory=list)
    language: list[str] = Field(default_factory=list)
    plot_summary: str = ""                           # text_info → 剧情简介
    themes_style: str = ""                           # 主题/风格 (chunk 后填充)
    review_highlights: str = ""                      # 评论摘要


class ChunkType:
    PLOT = "plot"
    THEMES_STYLE = "themes_style"
    REVIEWS = "reviews"


class Chunk(BaseModel):
    """单个文本块及其元数据。"""
    chunk_id: str                                    # {movie_id}_{chunk_type}
    movie_id: str
    chunk_type: str                                  # plot / themes_style / reviews
    text: str
    title_zh: str = ""
    year: int | None = None
    genres: list[str] = Field(default_factory=list)
    director: list[str] = Field(default_factory=list)
    rating_douban: float | None = None


class FilterConstraints(BaseModel):
    """结构化过滤条件。"""
    year_min: int | None = None
    year_max: int | None = None
    genres: list[str] | None = None
    rating_min: float | None = None
    countries: list[str] | None = None
    directors: list[str] | None = None


class ParsedQuery(BaseModel):
    """LLM 解析后的结构化查询。"""
    semantic_query: str                              # 用于语义检索的部分
    filters: FilterConstraints = Field(default_factory=FilterConstraints)
    similar_to: list[str] = Field(default_factory=list)  # 用户引用的电影名


class RetrievalResult(BaseModel):
    """单路检索结果。"""
    movie_id: str
    chunk_id: str
    score: float
    source: str                                      # vector / bm25 / metadata / similar_to
    rank: int = 0                                    # 在该路检索中的排名


class RankedMovie(BaseModel):
    """精排后的电影结果。"""
    movie_id: str
    score: float
    title_zh: str = ""
    year: int | None = None
    genres: list[str] = Field(default_factory=list)
    director: list[str] = Field(default_factory=list)
    rating_douban: float | None = None
    chunks: list[Chunk] = Field(default_factory=list)  # 召回到的 chunk
    explanation: str = ""


class Recommendation(BaseModel):
    """最终推荐结果。"""
    query: str
    parsed_query: ParsedQuery
    movies: list[RankedMovie]


class LabeledQuery(BaseModel):
    """一条评估标注。"""
    query: str
    relevant_movie_ids: list[str]
    source: str = "manual"                           # manual / douban_list / llm_generated


class EvalReport(BaseModel):
    """评估报告。"""
    version: str
    recall_at_10: float
    recall_at_50: float
    ndcg_at_10: float
    mrr: float
    num_queries: int
    notes: str = ""
