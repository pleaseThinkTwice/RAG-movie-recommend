"""数据管道编排器：加载 HuggingFace 数据集 → 清洗 → 过滤 → 输出 movies.jsonl。"""

import json
import os
from pathlib import Path
from typing import Optional

from tqdm import tqdm
from datasets import load_dataset

from src.core.config import get_config, Config
from src.core.schemas import Movie
from src.data.cleaner import (
    clean_text, extract_year_from_date,
    parse_separated_field, extract_runtime_minutes
)
from src.data.filters import (
    has_valid_comment, has_valid_score_count, has_valid_plot
)


def _get_cell(row: dict, *keys: str) -> Optional[str]:
    """尝试多个可能的列名获取值。"""
    for k in keys:
        val = row.get(k)
        if val is not None:
            return str(val) if not isinstance(val, str) else val
    return None


def load_raw_dataset() -> list[dict]:
    """从 HuggingFace 加载原始数据集，转为字典列表。"""
    print("Loading dataset from HuggingFace 'MangoGoes/douban_movie_info'...")
    ds = load_dataset("MangoGoes/douban_movie_info", split="train")
    print(f"  Total raw records: {len(ds)}")

    # 转为列表（方便迭代和处理）
    data = []
    for i in tqdm(range(len(ds)), desc="  Converting to list"):
        data.append(ds[i])
    return data


def row_to_movie(row: dict) -> Movie:
    """将原始行映射为 Movie 对象。"""

    # 电影名
    title_zh = str(row.get("movie_name", "")).strip()

    # IMDB ID 作为主键，否则用 douban movie_id
    imdb_id = row.get("IMDb", "")
    movie_id = str(imdb_id).strip() if imdb_id and str(imdb_id).strip() else f"douban_{row.get('movie_id', '')}"

    # 年代——从上映日期提取
    release_date = _get_cell(row, "上映日期")
    year = extract_year_from_date(release_date)

    # 类型 (通过列名匹配：包含"类"和"型"的那个列)
    genres_raw = _get_cell(row, "类型") or ""
    genres = parse_separated_field(genres_raw)

    # 导演
    directors_raw = str(row.get("director_names", ""))
    directors = parse_separated_field(directors_raw)

    # 演员
    actors_raw = str(row.get("actor_names", ""))
    # 只取前 10 位
    cast = parse_separated_field(actors_raw)[:10]

    # 评分
    score = row.get("score")
    rating_douban = float(score) if score else None
    score_amt = row.get("score_amt")
    rating_count = int(float(score_amt)) if score_amt else None

    # 片长
    runtime_str = _get_cell(row, "片长")
    runtime = extract_runtime_minutes(runtime_str)

    # 国家/地区
    country_raw = _get_cell(row, "制片国家/地区") or ""
    country = parse_separated_field(country_raw)

    # 语言
    language_raw = _get_cell(row, "语言") or ""
    language = parse_separated_field(language_raw)

    # 剧情简介 — 从 text_info 解析
    text_info = clean_text(str(row.get("text_info", "")))

    # 评论
    comment = clean_text(str(row.get("comment", "")))

    # 别名
    aka = _get_cell(row, "又名") or ""

    # 英文名（从别名或 text_info 提取）
    title_en = ""
    if aka:
        # 别名通常包含英文名
        for part in parse_separated_field(aka):
            if any(c.isascii() and c.isalpha() for c in part):
                title_en = part.strip()
                break

    return Movie(
        movie_id=movie_id,
        title_zh=title_zh,
        title_en=title_en,
        year=year,
        genres=genres,
        director=directors,
        cast=cast,
        rating_douban=rating_douban,
        rating_count=rating_count,
        runtime=runtime,
        country=country,
        language=language,
        plot_summary=text_info,
        themes_style="",         # Phase 2 填充
        review_highlights=comment,
    )


def run_pipeline(config: Optional[Config] = None, limit: int | None = None) -> list[Movie]:
    """完整数据管道：加载 → 映射 → 清洗 → 过滤 → 去重 → 输出。

    Args:
        config: 配置对象，默认加载全局配置。
        limit: 限制处理的电影数量（调试用），None 表示全部。

    Returns:
        过滤后的电影列表。
    """
    if config is None:
        config = get_config()

    raw = load_raw_dataset()
    if limit:
        raw = raw[:limit]

    cfg_filter = config.data.filter

    movies: list[Movie] = []
    seen_ids: set[str] = set()
    stats = {"total": len(raw), "mapped": 0, "passed": 0, "no_comment": 0,
             "no_score": 0, "no_plot": 0, "dup": 0}

    for row in tqdm(raw, desc="Processing movies"):
        try:
            movie = row_to_movie(row)
        except Exception:
            continue
        stats["mapped"] += 1

        # 过滤
        if not has_valid_comment(movie.review_highlights, cfg_filter.min_comment_len):
            stats["no_comment"] += 1
            continue
        if not has_valid_score_count(movie.rating_count, cfg_filter.min_score_count):
            stats["no_score"] += 1
            continue
        if not has_valid_plot(movie.plot_summary, cfg_filter.min_text_info_len):
            stats["no_plot"] += 1
            continue

        # 去重
        mid = movie.movie_id
        if mid in seen_ids:
            stats["dup"] += 1
            continue
        seen_ids.add(mid)

        movies.append(movie)
        stats["passed"] += 1

    # 输出
    os.makedirs(config.data.processed_dir, exist_ok=True)
    output_path = config.data.movies_file
    with open(output_path, "w", encoding="utf-8") as f:
        for m in movies:
            f.write(m.model_dump_json(ensure_ascii=False) + "\n")

    print(f"\nPipeline complete:")
    print(f"  Raw:       {stats['total']}")
    print(f"  Mapped:    {stats['mapped']}")
    print(f"  Passed:    {stats['passed']}")
    print(f"  Dropped:")
    print(f"    - no comment:  {stats['no_comment']}")
    print(f"    - no score:    {stats['no_score']}")
    print(f"    - no plot:     {stats['no_plot']}")
    print(f"    - duplicate:   {stats['dup']}")
    print(f"  Output: {output_path}")

    return movies
