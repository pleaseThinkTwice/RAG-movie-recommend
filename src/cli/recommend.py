"""CLI: 单条查询推荐（端到端 pipeline）。"""

import json
import sys
import io
import os
from pathlib import Path

from src.core.config import get_config
from src.core.schemas import Movie, Chunk, RankedMovie
from src.query.parser import QueryParser
from src.retrieval.orchestrator import RecallOrchestrator
from src.ranking.reranker import Reranker
from src.generation.explainer import Explainer


def _load_movies_dict(movies_path: str) -> dict[str, Movie]:
    """加载 movie_id → Movie 映射。"""
    movies = {}
    if os.path.exists(movies_path):
        with open(movies_path, "r", encoding="utf-8") as f:
            for line in f:
                m = Movie.model_validate_json(line.strip())
                movies[m.movie_id] = m
    return movies


def _load_chunks_dict(chunks_path: str) -> dict[str, list[Chunk]]:
    """加载 movie_id → [Chunk] 映射。"""
    chunks_map: dict[str, list[Chunk]] = {}
    chunk_by_id: dict[str, Chunk] = {}
    if os.path.exists(chunks_path):
        with open(chunks_path, "r", encoding="utf-8") as f:
            for line in f:
                c = Chunk.model_validate_json(line.strip())
                if c.movie_id not in chunks_map:
                    chunks_map[c.movie_id] = []
                chunks_map[c.movie_id].append(c)
                chunk_by_id[c.chunk_id] = c
    return chunks_map


def recommend(
    user_query: str,
    verbose: bool = True,
) -> list[RankedMovie]:
    """端到端推荐 pipeline。

    Returns:
        Top-10 推荐电影列表（含解释）。
    """
    config = get_config()

    # 1. Query 解析
    if verbose:
        print(f"Query: {user_query}")
        print("Parsing query...")
    parser = QueryParser()
    parsed = parser.parse_sync(user_query)
    if verbose:
        print(f"  semantic_query: {parsed.semantic_query}")
        print(f"  filters: {parsed.filters.model_dump(exclude_none=True)}")
        print(f"  similar_to: {parsed.similar_to}")

    # 2. 召回
    if verbose:
        print("Recalling candidates...")
    orchestrator = RecallOrchestrator()
    recall_results = orchestrator.recall(parsed)
    if verbose:
        print(f"  Recalled {len(recall_results)} candidates (unique movies: {len(set(r.movie_id for r in recall_results))})")

    # 3. 精排
    if verbose:
        print("Reranking...")
    # 把 recall_results 聚合成 (chunk, score) 对
    chunks_map = _load_chunks_dict(config.data.chunks_file)
    chunk_by_id: dict[str, Chunk] = {}
    for clist in chunks_map.values():
        for c in clist:
            chunk_by_id[c.chunk_id] = c

    reranker = Reranker()
    # 去重到 movie 级别，每部电影取最高分 chunk
    movie_best: dict[str, tuple[Chunk, float]] = {}
    for r in recall_results:
        if r.chunk_id not in chunk_by_id:
            continue
        chunk = chunk_by_id[r.chunk_id]
        if r.movie_id not in movie_best or r.score > movie_best[r.movie_id][1]:
            movie_best[r.movie_id] = (chunk, r.score)

    sorted_movies: list[tuple[Chunk, float]] = []
    if movie_best:
        sorted_movies = reranker.rerank(
            parsed.semantic_query,
            [c for c, _ in movie_best.values()],
            top_k=10,
        )

    # 4. 生成解释
    if verbose:
        print("Generating explanations...")
    movies_dict = _load_movies_dict(config.data.movies_file)
    explainer = Explainer()

    results: list[RankedMovie] = []
    for chunk, score in sorted_movies:
        movie = movies_dict.get(chunk.movie_id)
        if not movie:
            continue

        # 获取该电影的所有 chunk
        movie_chunks = chunks_map.get(chunk.movie_id, [chunk])

        # 生成解释
        explanation = explainer.explain(user_query, movie.title_zh, movie_chunks)

        results.append(RankedMovie(
            movie_id=movie.movie_id,
            score=float(score),
            title_zh=movie.title_zh,
            year=movie.year,
            genres=movie.genres,
            director=movie.director,
            rating_douban=movie.rating_douban,
            chunks=movie_chunks,
            explanation=explanation,
        ))

    return results


def main():
    if sys.stdout.encoding != 'utf-8':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    import argparse
    ap = argparse.ArgumentParser(description="RAG Movie Recommendation")
    ap.add_argument("query", nargs="?", default="烧脑科幻大片",
                    help="Natural language query for movie recommendation")
    ap.add_argument("--no-verbose", action="store_true", help="Suppress progress output")
    args = ap.parse_args()

    results = recommend(args.query, verbose=not args.no_verbose)

    print(f"\n{'='*60}")
    print(f"  Top-{len(results)} Recommendations")
    print(f"{'='*60}\n")

    for i, r in enumerate(results, 1):
        print(f"{i}. 《{r.title_zh}》({r.year}) {r.rating_douban}/10")
        print(f"   导演: {'/'.join(r.director) if r.director else '未知'}")
        print(f"   类型: {'/'.join(r.genres) if r.genres else '未知'}")
        print(f"   推荐理由: {r.explanation}")
        print()


if __name__ == "__main__":
    main()
