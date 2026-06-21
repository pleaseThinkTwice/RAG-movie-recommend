"""生成层：基于检索 chunk 生成推荐理由，强制 Grounding。"""

import asyncio
import json
from typing import Optional

from openai import OpenAI

from src.core.config import get_config
from src.core.schemas import Movie, Chunk

EXPLAIN_PROMPT = """你是一个电影推荐助手。用户的需求是：{user_query}

候选电影：《{title}》
相关信息：
- 剧情：{plot_text}
- 主题风格：{themes_text}
- 用户评论：{reviews_text}

请用2-3句话说明这部电影为什么符合用户需求，要求：
1. 必须基于上面提供的信息，不要编造剧情、演员、导演等事实。
2. 必须明确指出符合用户哪个需求点。
3. 如果某个需求点没法从信息中确认，直接说"具体是否符合XX需要观看后判断"，不要瞎编。

只输出推荐理由，不要加前缀或引号。"""


class Explainer:
    """为推荐电影生成有依据的解释。"""

    def __init__(self, model: str | None = None, base_url: str | None = None, api_key: str | None = None):
        config = get_config().llm
        self.model = model or config.model
        self.base_url = base_url or config.base_url
        self.api_key = api_key or config.api_key
        self.temperature = config.temperature_generation
        self.max_tokens = config.max_tokens_generation

        self.client = OpenAI(
            api_key=self.api_key or "dummy",
            base_url=self.base_url,
        )

    def explain(
        self,
        user_query: str,
        title: str,
        chunks: list[Chunk],
    ) -> str:
        """为一部电影生成推荐理由（同步）。"""
        # 按类型提取 chunk 文本
        plot_text = ""
        themes_text = ""
        reviews_text = ""

        for c in chunks:
            if c.chunk_type == "plot":
                plot_text = c.text
            elif c.chunk_type == "themes_style":
                themes_text = c.text
            elif c.chunk_type == "reviews":
                reviews_text = c.text

        prompt = EXPLAIN_PROMPT.format(
            user_query=user_query,
            title=title,
            plot_text=plot_text or "暂无剧情信息",
            themes_text=themes_text or "暂无主题风格信息",
            reviews_text=reviews_text or "暂无评论信息",
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Explanation generation failed: {e}")
            return f"这部电影在类型和风格上与您的需求匹配，推荐您观看。"

    async def explain_async(self, user_query: str, title: str, chunks: list[Chunk]) -> str:
        """异步版本。"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.explain, user_query, title, chunks)

    async def explain_batch(
        self,
        user_query: str,
        movies_with_chunks: list[tuple[str, list[Chunk]]],  # [(title, chunks), ...]
    ) -> list[str]:
        """并发为多部电影生成解释。"""
        tasks = [
            self.explain_async(user_query, title, chunks)
            for title, chunks in movies_with_chunks
        ]
        return await asyncio.gather(*tasks, return_exceptions=True)
