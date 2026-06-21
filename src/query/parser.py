"""LLM Query 结构化解析：将自然语言查询拆分为语义部分 + 过滤条件 + 参照电影。"""

import json
import re
from typing import Optional

from openai import OpenAI

from src.core.config import get_config
from src.core.schemas import ParsedQuery, FilterConstraints

QUERY_PARSE_PROMPT = """你是一个电影推荐系统的query解析助手。请把用户的自然语言query拆解为结构化JSON：

输出schema:
{
  "semantic_query": "用于语义检索的部分，去除硬约束后的纯感受/风格描述",
  "filters": {
    "year_min": 最小年份(nullable),
    "year_max": 最大年份(nullable),
    "genres": ["类型列表，必须从常见电影类型中选"],
    "rating_min": 最低评分(nullable),
    "countries": ["国家/地区列表"],
    "directors": ["导演名列表"]
  },
  "similar_to": ["用户引用的电影名，用于查找风格相似的电影"]
}

约束：
- 不要凭空生成filter，用户没明说的字段填null。
- semantic_query要保留用户的感受性表达（如"烧脑""治愈""慢节奏"），去掉硬约束词（如"2010年以后的""评分7分以上"）。
- 只输出JSON，不要任何解释。

User query: {user_input}"""


class QueryParser:
    """使用 LLM 将自然语言 query 解析为结构化 ParsedQuery。"""

    def __init__(self, model: str | None = None, base_url: str | None = None, api_key: str | None = None):
        config = get_config().llm
        self.model = model or config.model
        self.base_url = base_url or config.base_url
        self.api_key = api_key or config.api_key
        self.temperature = config.temperature_query_parse
        self._cache: dict[str, ParsedQuery] = {}

        if not self.api_key:
            print("WARNING: No LLM API key set. Query parsing will use fallback mode.")

        self.client = OpenAI(
            api_key=self.api_key or "dummy",
            base_url=self.base_url,
        )

    async def parse(self, user_query: str) -> ParsedQuery:
        """解析用户查询。"""
        # 缓存命中
        cache_key = user_query.strip()
        if cache_key in self._cache:
            return self._cache[cache_key]

        # 尝试 LLM 解析
        try:
            result = await self._call_llm(cache_key)
            self._cache[cache_key] = result
            return result
        except Exception as e:
            print(f"Query parse failed, using fallback: {e}")
            fallback = self._fallback_parse(cache_key)
            self._cache[cache_key] = fallback
            return fallback

    def parse_sync(self, user_query: str) -> ParsedQuery:
        """同步版本的解析（用于简单场景）。"""
        cache_key = user_query.strip()
        if cache_key in self._cache:
            return self._cache[cache_key]

        try:
            prompt = QUERY_PARSE_PROMPT.format(user_input=cache_key)
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                response_format={"type": "json_object"},
                max_tokens=512,
            )
            raw = response.choices[0].message.content
            result = self._parse_response(raw)
            self._cache[cache_key] = result
            return result
        except Exception as e:
            print(f"Query parse failed, using fallback: {e}")
            fallback = self._fallback_parse(cache_key)
            self._cache[cache_key] = fallback
            return fallback

    async def _call_llm(self, user_query: str) -> ParsedQuery:
        """调用 LLM 进行解析。"""
        prompt = QUERY_PARSE_PROMPT.format(user_input=user_query)
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            response_format={"type": "json_object"},
            max_tokens=512,
        )
        raw = response.choices[0].message.content
        return self._parse_response(raw)

    def _parse_response(self, raw: str | None) -> ParsedQuery:
        """解析 LLM 返回的 JSON。"""
        if not raw:
            return self._fallback_parse("")

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            # 尝试提取 JSON 块
            match = re.search(r'\{.*\}', raw, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group())
                except json.JSONDecodeError:
                    return self._fallback_parse(raw)
            else:
                return self._fallback_parse(raw)

        filters_raw = data.get("filters", {}) or {}
        filters = FilterConstraints(
            year_min=filters_raw.get("year_min"),
            year_max=filters_raw.get("year_max"),
            genres=filters_raw.get("genres"),
            rating_min=filters_raw.get("rating_min"),
            countries=filters_raw.get("countries"),
            directors=filters_raw.get("directors"),
        )

        semantic_query = data.get("semantic_query", "").strip()
        if not semantic_query:
            semantic_query = data.get("user_input", "")

        similar_to = data.get("similar_to") or []

        return ParsedQuery(
            semantic_query=semantic_query,
            filters=filters,
            similar_to=similar_to,
        )

    def _fallback_parse(self, user_query: str) -> ParsedQuery:
        """降级方案：整个 query 作为 semantic_query，filters 为空。"""
        return ParsedQuery(
            semantic_query=user_query,
            filters=FilterConstraints(),
            similar_to=[],
        )
