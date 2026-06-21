"""结构化 Chunking：每部电影切为 plot / themes_style / reviews 三个 chunk。"""

import json
import os
from pathlib import Path
from typing import Optional

import jieba
import jieba.analyse
from tqdm import tqdm

from src.core.config import get_config, Config
from src.core.schemas import Movie, Chunk, ChunkType


def load_style_dict(config: Optional[Config] = None) -> dict:
    """加载风格词典。"""
    if config is None:
        config = get_config()
    path = config.data.style_dict
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_keywords_tfidf(text: str, top_k: int = 10) -> list[str]:
    """用 jieba TF-IDF 从文本提取关键词。"""
    if not text.strip():
        return []
    # jieba TF-IDF 关键词提取
    keywords = jieba.analyse.extract_tags(text, topK=top_k)
    return keywords


def match_style_tags(text: str, style_dict: dict | None = None) -> list[str]:
    """用风格词典匹配文本中的风格/观感标签。"""
    if style_dict is None:
        style_dict = load_style_dict()
    if not text or not style_dict:
        return []

    matched = set()
    # 确保 jieba 已加载默认词典
    text_lower = text.lower()

    for category, tags_dict in style_dict.items():
        if not isinstance(tags_dict, dict):
            continue
        for tag_name, variants in tags_dict.items():
            if not isinstance(variants, list):
                continue
            for variant in variants:
                if variant.lower() in text_lower:
                    matched.add(tag_name)
                    break

    return list(matched)


def build_themes_style_text(movie: Movie, style_dict: dict | None = None) -> str:
    """构建 themes_style chunk 的文本内容。"""
    parts = []

    # 1. 类型标签
    if movie.genres:
        parts.append(f"[类型] {' / '.join(movie.genres)}")

    # 2. TF-IDF 关键词
    combined_text = f"{movie.plot_summary} {movie.review_highlights}"
    keywords = extract_keywords_tfidf(combined_text, top_k=10)
    if keywords:
        parts.append(f"[关键词] {' '.join(keywords)}")

    # 3. 风格标签
    styles = match_style_tags(movie.review_highlights, style_dict)
    if not styles and movie.plot_summary:
        styles = match_style_tags(movie.plot_summary, style_dict)
    if styles:
        parts.append(f"[风格] {' '.join(styles)}")

    return "\n".join(parts)


def chunk_movie(movie: Movie, style_dict: dict | None = None) -> list[Chunk]:
    """将一部电影切分为 3 个结构化 chunk。"""
    chunks = []

    # --- plot chunk ---
    plot_text = movie.plot_summary
    if not plot_text.strip():
        plot_text = f"{movie.title_zh} - 暂无剧情简介"
    chunks.append(Chunk(
        chunk_id=f"{movie.movie_id}_plot",
        movie_id=movie.movie_id,
        chunk_type=ChunkType.PLOT,
        text=plot_text,
        title_zh=movie.title_zh,
        year=movie.year,
        genres=movie.genres,
        director=movie.director,
        rating_douban=movie.rating_douban,
    ))

    # --- themes_style chunk ---
    themes_text = build_themes_style_text(movie, style_dict)
    if not themes_text.strip():
        themes_text = f"{movie.title_zh} - {' / '.join(movie.genres)}" if movie.genres else movie.title_zh
    chunks.append(Chunk(
        chunk_id=f"{movie.movie_id}_themes_style",
        movie_id=movie.movie_id,
        chunk_type=ChunkType.THEMES_STYLE,
        text=themes_text,
        title_zh=movie.title_zh,
        year=movie.year,
        genres=movie.genres,
        director=movie.director,
        rating_douban=movie.rating_douban,
    ))

    # --- reviews chunk ---
    review_text = movie.review_highlights
    if not review_text.strip():
        review_text = f"{movie.title_zh} - 暂无评论"
    else:
        # 格式化评论（用 | 分隔多条评论）
        review_parts = review_text.split("|")
        review_parts = [f"评论{rp.strip()}" for rp in review_parts[:5] if rp.strip()]
        review_text = "\n".join(review_parts) if review_parts else review_text
    chunks.append(Chunk(
        chunk_id=f"{movie.movie_id}_reviews",
        movie_id=movie.movie_id,
        chunk_type=ChunkType.REVIEWS,
        text=review_text,
        title_zh=movie.title_zh,
        year=movie.year,
        genres=movie.genres,
        director=movie.director,
        rating_douban=movie.rating_douban,
    ))

    return chunks


def run_chunking(config: Optional[Config] = None) -> list[Chunk]:
    """对 movies.jsonl 中的所有电影执行结构化 chunking。"""
    if config is None:
        config = get_config()

    style_dict = load_style_dict(config)
    print(f"Loaded style dict with {sum(len(tags) for tags in style_dict.values())} tags")

    # 加载电影
    movies_path = config.data.movies_file
    movies: list[Movie] = []
    with open(movies_path, "r", encoding="utf-8") as f:
        for line in f:
            movies.append(Movie.model_validate_json(line.strip()))
    print(f"Loaded {len(movies)} movies from {movies_path}")

    # 切分
    all_chunks: list[Chunk] = []
    for movie in tqdm(movies, desc="Chunking movies"):
        chunks = chunk_movie(movie, style_dict)
        all_chunks.extend(chunks)

    # 保存
    chunks_path = config.data.chunks_file
    os.makedirs(os.path.dirname(chunks_path), exist_ok=True)
    with open(chunks_path, "w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(c.model_dump_json(ensure_ascii=False) + "\n")

    print(f"Chunking complete: {len(movies)} movies → {len(all_chunks)} chunks")
    print(f"  - plot:         {sum(1 for c in all_chunks if c.chunk_type == 'plot')}")
    print(f"  - themes_style: {sum(1 for c in all_chunks if c.chunk_type == 'themes_style')}")
    print(f"  - reviews:      {sum(1 for c in all_chunks if c.chunk_type == 'reviews')}")
    print(f"  Output: {chunks_path}")

    return all_chunks
