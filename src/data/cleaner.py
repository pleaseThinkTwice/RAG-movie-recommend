"""文本清洗：去 HTML、统一标点、折叠重复字符。"""

import re
import unicodedata


def strip_html(text: str) -> str:
    """去除 HTML 标签。"""
    import re as _re
    return _re.sub(r'<[^>]+>', '', text)


def normalize_punctuation(text: str) -> str:
    """全角标点转半角，统一 Unicode 归一化 (NFKC)。"""
    return unicodedata.normalize('NFKC', text)


def collapse_repeated_chars(text: str) -> str:
    """折叠重复字符，如 '哈哈哈哈哈' → '哈哈'。"""
    # 中文字符重复
    text = re.sub(r'([一-鿿])\1{2,}', r'\1\1', text)
    # 常见重复语气词
    text = re.sub(r'([哈呵嘿嗯哦])\1{1,}', r'\1\1', text)
    # 标点重复
    text = re.sub(r'([!！?？。.])\1{2,}', r'\1\1', text)
    return text


def clean_text(text: str | None) -> str:
    """完整清洗流程。"""
    if not text:
        return ""
    text = strip_html(text)
    text = normalize_punctuation(text)
    text = collapse_repeated_chars(text)
    return text.strip()


def extract_year_from_date(date_str: str | None) -> int | None:
    """从日期字符串中提取年份。支持多种格式：
    '1987-10-04(东京国际电影节)' → 1987
    '1987' → 1987
    """
    if not date_str:
        return None
    match = re.search(r'(\d{4})', str(date_str))
    return int(match.group(1)) if match else None


def parse_separated_field(value: str | None, sep: str = "/") -> list[str]:
    """解析用分隔符分隔的字段，如 '英国/意大利/中国大陆' → ['英国', '意大利', '中国大陆']。"""
    if not value:
        return []
    return [item.strip() for item in str(value).split(sep) if item.strip()]


def extract_runtime_minutes(runtime_str: str | None) -> int | None:
    """从片长字符串提取分钟数。'163分钟/210分钟(加长版)' → 163。"""
    if not runtime_str:
        return None
    match = re.search(r'(\d+)\s*分钟', str(runtime_str))
    return int(match.group(1)) if match else None
