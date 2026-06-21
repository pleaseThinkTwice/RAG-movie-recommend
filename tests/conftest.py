"""共享测试夹具。"""

import pytest

from src.core.schemas import (
    Movie, Chunk, ParsedQuery, FilterConstraints, RetrievalResult
)


@pytest.fixture
def sample_movies() -> list[Movie]:
    """5 部手工电影，覆盖不同 chunk 类型场景。"""
    return [
        Movie(
            movie_id="tt1375666",
            title_zh="盗梦空间",
            title_en="Inception",
            year=2010,
            genres=["科幻", "动作", "悬疑"],
            director=["克里斯托弗·诺兰"],
            cast=["莱昂纳多·迪卡普里奥", "渡边谦"],
            rating_douban=9.4,
            rating_count=1850000,
            runtime=148,
            country=["美国", "英国"],
            language=["英语", "日语"],
            plot_summary="道姆·柯布是一位经验老道的窃贼，他能潜入人们梦境中盗取潜意识中最有价值的秘密。为了回到孩子们身边，他接受了一项看似不可能的任务——在目标人物的潜意识中植入一个想法。",
            themes_style="",
            review_highlights="神作 反转 梦境 烧脑 多层结构 视觉奇观"
        ),
        Movie(
            movie_id="tt0816692",
            title_zh="星际穿越",
            title_en="Interstellar",
            year=2014,
            genres=["科幻", "冒险", "剧情"],
            director=["克里斯托弗·诺兰"],
            cast=["马修·麦康纳", "安妮·海瑟薇"],
            rating_douban=9.4,
            rating_count=1450000,
            runtime=169,
            country=["美国", "英国"],
            language=["英语"],
            plot_summary="在不远的未来，地球环境急剧恶化。前NASA宇航员库珀被选中驾驶宇宙飞船穿越虫洞，为人类寻找新的家园。",
            themes_style="",
            review_highlights="硬科幻 父女情 催泪 黑洞 时间膨胀 视觉震撼"
        ),
        Movie(
            movie_id="tt0133093",
            title_zh="黑客帝国",
            title_en="The Matrix",
            year=1999,
            genres=["科幻", "动作"],
            director=["沃卓斯基姐妹"],
            cast=["基努·里维斯", "劳伦斯·菲什伯恩"],
            rating_douban=9.1,
            rating_count=750000,
            runtime=136,
            country=["美国"],
            language=["英语"],
            plot_summary="程序员尼奥发现看似正常的现实世界其实是由一个名为矩阵的计算机人工智能系统控制的虚拟世界。",
            themes_style="",
            review_highlights="赛博朋克 哲学 红蓝药丸 子弹时间 经典 颠覆认知"
        ),
        Movie(
            movie_id="tt2582802",
            title_zh="爆裂鼓手",
            title_en="Whiplash",
            year=2014,
            genres=["剧情", "音乐"],
            director=["达米恩·查泽雷"],
            cast=["迈尔斯·特勒", "J·K·西蒙斯"],
            rating_douban=8.7,
            rating_count=520000,
            runtime=106,
            country=["美国"],
            language=["英语"],
            plot_summary="19岁的安德鲁是一个有抱负的爵士鼓手，他在严苛的音乐学院遇到了魔鬼导师弗莱彻。弗莱彻以极端的方式逼迫学生突破极限。",
            themes_style="",
            review_highlights="热血 偏执 师徒 节奏 极限 黑暗 励志"
        ),
        Movie(
            movie_id="tt10272386",
            title_zh="困在时间里的父亲",
            title_en="The Father",
            year=2020,
            genres=["剧情"],
            director=["佛罗莱恩·泽勒"],
            cast=["安东尼·霍普金斯", "奥利维娅·科尔曼"],
            rating_douban=8.8,
            rating_count=280000,
            runtime=97,
            country=["英国", "法国"],
            language=["英语"],
            plot_summary="年迈的安东尼拒绝女儿的帮助，但他发现自己逐渐分不清现实与幻觉。",
            themes_style="",
            review_highlights="阿兹海默 压抑 视角切换 演技炸裂 催泪 烧脑叙事"
        ),
    ]


@pytest.fixture
def sample_chunks(sample_movies) -> list[Chunk]:
    """为 sample_movies 生成结构化 chunk。"""
    chunks = []
    for m in sample_movies:
        chunks.append(Chunk(
            chunk_id=f"{m.movie_id}_plot",
            movie_id=m.movie_id,
            chunk_type="plot",
            text=m.plot_summary,
            title_zh=m.title_zh, year=m.year,
            genres=m.genres, director=m.director,
            rating_douban=m.rating_douban
        ))
        chunks.append(Chunk(
            chunk_id=f"{m.movie_id}_themes_style",
            movie_id=m.movie_id,
            chunk_type="themes_style",
            text=f"类型: {' / '.join(m.genres)}\n关键词: 烧脑 非线性叙事",
            title_zh=m.title_zh, year=m.year,
            genres=m.genres, director=m.director,
            rating_douban=m.rating_douban
        ))
        chunks.append(Chunk(
            chunk_id=f"{m.movie_id}_reviews",
            movie_id=m.movie_id,
            chunk_type="reviews",
            text=m.review_highlights,
            title_zh=m.title_zh, year=m.year,
            genres=m.genres, director=m.director,
            rating_douban=m.rating_douban
        ))
    return chunks


@pytest.fixture
def sample_parsed_query() -> ParsedQuery:
    return ParsedQuery(
        semantic_query="烧脑科幻大片",
        filters=FilterConstraints(year_min=2010, genres=["科幻"]),
        similar_to=[]
    )
