"""LLM Query 结构化解析：将自然语言查询拆分为语义部分 + 过滤条件 + 参照电影。"""

import json
import re
from typing import Optional

from openai import OpenAI

from src.core.config import get_config
from src.core.schemas import ParsedQuery, FilterConstraints

QUERY_PARSE_PROMPT = """你是一个电影推荐系统的query解析助手。请把用户的自然语言query拆解为JSON。

输出schema:
{{
  "semantic_query": "用于语义检索的部分，去除硬约束后的纯感受/风格描述",
  "filters": {{
    "year_min": null,
    "year_max": null,
    "genres": null,
    "rating_min": null,
    "countries": null,
    "directors": null
  }},
  "similar_to": null
}}

规则：
- 不要凭空生成filter，用户没明说的字段保持null。
- semantic_query要保留用户的感受性表达（如"烧脑""治愈""慢节奏"），去掉硬约束词（如"2010年以后的""评分7分以上"）。
- 必须输出合法JSON，不要任何其他文字。

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
        self._cache_path = "data/processed/query_cache.jsonl"

        # 加载磁盘缓存
        self._load_disk_cache()

        if not self.api_key:
            print("WARNING: No LLM API key set. Query parsing will use fallback mode.")

        self.client = OpenAI(
            api_key=self.api_key or "dummy",
            base_url=self.base_url,
        )

    def _load_disk_cache(self):
        """从 JSONL 文件加载缓存的解析结果。"""
        try:
            if os.path.exists(self._cache_path):
                with open(self._cache_path, "r", encoding="utf-8") as f:
                    for line in f:
                        entry = json.loads(line.strip())
                        pq = ParsedQuery.model_validate(entry["parsed"])
                        self._cache[entry["query"]] = pq
        except Exception:
            pass

    def _save_to_disk_cache(self, query: str, parsed: ParsedQuery):
        """追加一条缓存到磁盘。"""
        try:
            os.makedirs(os.path.dirname(self._cache_path), exist_ok=True)
            with open(self._cache_path, "a", encoding="utf-8") as f:
                entry = {"query": query, "parsed": parsed.model_dump()}
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception:
            pass

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
                max_tokens=512,
            )
            raw = response.choices[0].message.content
            result = self._parse_response(raw)
            self._cache[cache_key] = result
            self._save_to_disk_cache(cache_key, result)
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
            # DeepSeek doesn't support response_format JSON mode well
            max_tokens=512,
        )
        raw = response.choices[0].message.content
        return self._parse_response(raw)

    def _parse_response(self, raw: str | None) -> ParsedQuery:
        """解析 LLM 返回的 JSON。支持多种格式。"""
        if not raw:
            return self._fallback_parse("")

        # 1. 直接解析
        try:
            data = json.loads(raw)
            return self._build_from_data(data)
        except json.JSONDecodeError:
            pass

        # 2. 提取 JSON 块 (支持 markdown ```json 格式)
        # 先去掉 markdown 代码块
        cleaned = re.sub(r'```(?:json)?\s*', '', raw)
        cleaned = re.sub(r'```\s*', '', cleaned)
        cleaned = cleaned.strip()

        try:
            data = json.loads(cleaned)
            return self._build_from_data(data)
        except json.JSONDecodeError:
            pass

        # 3. 提取第一个 { } 块
        match = re.search(r'\{.*\}', cleaned, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
                return self._build_from_data(data)
            except json.JSONDecodeError:
                pass

        return self._fallback_parse(raw)

    def _build_from_data(self, data: dict) -> ParsedQuery:
        """从解析的 JSON 字典构建 ParsedQuery。"""
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
            semantic_query = " ".join([
                str(data.get("filters", {}).get("genres", "")),
                str(data.get("similar_to", ""))
            ]).strip() or ""

        similar_to = data.get("similar_to") or []
        # 兼容 LLM 返回字符串而非列表的情况
        if isinstance(similar_to, str):
            similar_to = [similar_to] if similar_to.strip() else []
        if not isinstance(similar_to, list):
            similar_to = []
        # 也处理 filters 中的字段类型问题
        if isinstance(filters.genres, str):
            filters.genres = [filters.genres] if filters.genres.strip() else None
        if isinstance(filters.countries, str):
            filters.countries = [filters.countries] if filters.countries.strip() else None
        if isinstance(filters.directors, str):
            filters.directors = [filters.directors] if filters.directors.strip() else None
        return ParsedQuery(
            semantic_query=semantic_query,
            filters=filters,
            similar_to=similar_to,
        )

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
