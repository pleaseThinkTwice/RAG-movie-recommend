"""Grounding 验证：检查 LLM 生成的解释是否能在 source chunk 中找到依据。"""

import re
from typing import Optional

from src.core.schemas import Chunk


def extract_claims(explanation: str) -> list[str]:
    """从解释文本中简单提取关键声明（按句号/逗号分句）。"""
    sentences = re.split(r'[，。,\.\n]', explanation)
    return [s.strip() for s in sentences if len(s.strip()) >= 5]


def verify_claim_in_chunks(claim: str, chunks: list[Chunk]) -> bool:
    """检查一个声明是否在 chunk 文本中有支撑。

    简易方法：检查 claim 中 >= 3 个字的子串是否在 chunk 中出现。
    """
    # 提取 3-gram（中文）
    trigrams = set()
    for i in range(len(claim) - 2):
        trigram = claim[i:i+3]
        if not any(c in '，。、！？；：""''（）\n\r\t ' for c in trigram):
            trigrams.add(trigram)

    if not trigrams:
        return True  # 太短的声明放行

    # 检查是否有足够比例的 trigram 在 chunk 中
    found = 0
    for tg in trigrams:
        for c in chunks:
            if tg in c.text:
                found += 1
                break

    return found / len(trigrams) >= 0.3  # 至少 30% 的 trigram 能在 chunk 中找到


def verify_grounding(
    explanation: str,
    chunks: list[Chunk],
) -> dict:
    """验证解释的 Grounding 质量。

    Returns:
        {
            "total_claims": int,
            "grounded_claims": int,
            "ungrounded_claims": list[str],
            "grounding_ratio": float,
        }
    """
    claims = extract_claims(explanation)
    if not claims:
        return {
            "total_claims": 0,
            "grounded_claims": 0,
            "ungrounded_claims": [],
            "grounding_ratio": 1.0,
        }

    grounded = []
    ungrounded = []
    for claim in claims:
        if verify_claim_in_chunks(claim, chunks):
            grounded.append(claim)
        else:
            ungrounded.append(claim)

    return {
        "total_claims": len(claims),
        "grounded_claims": len(grounded),
        "ungrounded_claims": ungrounded,
        "grounding_ratio": len(grounded) / len(claims) if claims else 1.0,
    }
