"""文本后处理:繁简转换等。"""

from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=1)
def _converter():
    try:
        from opencc import OpenCC

        return OpenCC("t2s")
    except Exception:  # opencc 未安装时静默降级
        return None


def to_simplified(text: str) -> str:
    """繁体转简体。Whisper 和 LLM 偶尔会混入繁体字,统一转简体。

    opencc 不可用时原样返回,不影响主流程。
    """
    if not text:
        return text
    cc = _converter()
    return cc.convert(text) if cc else text


def segments_to_simplified(segments: list[dict]) -> list[dict]:
    """对分段列表逐段做繁简转换。"""
    return [{**s, "text": to_simplified(s.get("text", ""))} for s in segments]


def deep_to_simplified(obj):
    """递归转换嵌套结构中的字符串(用于 LLM 摘要等 JSON)。"""
    if isinstance(obj, str):
        return to_simplified(obj)
    if isinstance(obj, list):
        return [deep_to_simplified(x) for x in obj]
    if isinstance(obj, dict):
        return {k: deep_to_simplified(v) for k, v in obj.items()}
    return obj
