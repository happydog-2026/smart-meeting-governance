"""翻译服务:百度翻译 API 优先,LLM 兜底。

- 未配置百度 key(BAIDU_TRANSLATE_APPID / BAIDU_TRANSLATE_KEY)时,直接走 LLM。
- 百度调用失败(网络 / 配额 / 签名)自动回退 LLM,保证功能始终可用。
- 结构化纪要采用「单次请求 + 换行分隔」翻译,省请求次数(百度免费档 QPS=1),
  字段数对不上时回退 LLM。

百度通用翻译接口约定:
    GET https://fanyi-api.baidu.com/api/trans/vip/translate
    签名 sign = md5(appid + q + salt + 密钥),q 为未 URL 编码的原始文本。
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import re
import urllib.parse
import urllib.request

from app.config import settings
from app.services import llm

logger = logging.getLogger(__name__)

BAIDU_ENDPOINT = "https://fanyi-api.baidu.com/api/trans/vip/translate"


def _baidu_available() -> bool:
    return bool(settings.baidu_appid and settings.baidu_key)


def _baidu_request(q: str, target: str) -> str:
    """单次调用百度通用翻译,返回译文字符串。"""
    appid, key = settings.baidu_appid, settings.baidu_key
    salt = str(random.randint(10000, 99999))
    sign = hashlib.md5((appid + q + salt + key).encode("utf-8")).hexdigest()
    params = {
        "q": q,
        "from": "auto",
        "to": target,
        "appid": appid,
        "salt": salt,
        "sign": sign,
    }
    url = BAIDU_ENDPOINT + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    if "error_code" in data:
        raise RuntimeError(f"百度翻译错误 {data['error_code']}: {data.get('error_msg')}")
    return "".join(item.get("dst", "") for item in data.get("trans_result", []))


def _split_chunks(text: str, max_chars: int = 1500) -> list[str]:
    """按句末标点切块,每块不超过 max_chars(百度单次限 6000 字节)。"""
    text = text.strip()
    if len(text) <= max_chars:
        return [text] if text else []
    chunks: list[str] = []
    start, n = 0, len(text)
    while start < n:
        end = min(start + max_chars, n)
        if end < n:
            for i in range(end, start + max_chars // 2, -1):
                if text[i - 1] in "。!?.;!?…\n":
                    end = i
                    break
        chunks.append(text[start:end].strip())
        start = end
    return chunks


def _parse_glossary(glossary: str) -> list[str]:
    """把术语表字符串拆成去重、非空的术语列表(逗号/顿号/分号/空白分隔)。"""
    if not glossary:
        return []
    terms: list[str] = []
    for raw in re.split(r"[,、;；\s]+", glossary.strip()):
        raw = raw.strip()
        if raw and raw not in terms:
            terms.append(raw)
    return terms


def _protect(text: str, terms: list[str]) -> tuple[str, dict[str, str]]:
    """用占位符替换术语,返回 (处理后文本, {占位符: 术语})。

    按术语长度降序替换,避免「AI」把「AIGC」切碎。
    """
    mapping: dict[str, str] = {}
    for i, term in enumerate(sorted(terms, key=len, reverse=True)):
        if term in text:
            ph = f"[[T{i}]]"
            text = text.replace(term, ph)
            mapping[ph] = term
    return text, mapping


def _restore(text: str, mapping: dict[str, str]) -> str:
    """把占位符还原为术语。"""
    for ph, term in mapping.items():
        text = text.replace(ph, term)
    return text


def _baidu_translate_text(text: str, target: str, glossary: str = "") -> str:
    """长文本分块翻译(逐块调用,结果以换行拼接);翻译前后用占位符保护术语。"""
    terms = _parse_glossary(glossary)
    out = []
    for chunk in _split_chunks(text):
        protected, mapping = _protect(chunk, terms)
        out.append(_restore(_baidu_request(protected, target), mapping))
    return "\n".join(out)


def translate(
    text: str,
    *,
    target: str = "en",
    glossary: str = "",
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> str:
    """翻译纯文本。百度优先,失败回退 LLM。model/api_key/base_url 仅用于 LLM 兜底。"""
    if not text.strip():
        return text
    if _baidu_available():
        try:
            return _baidu_translate_text(text, target, glossary)
        except Exception as exc:  # pragma: no cover - 网络/配额相关
            logger.warning("百度翻译失败,回退 LLM: %s", exc)
    return llm.translate(
        text, target=target, glossary=glossary, model=model, api_key=api_key, base_url=base_url
    )


def translate_summary(
    summary: dict,
    *,
    target: str = "en",
    glossary: str = "",
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> dict:
    """翻译结构化纪要(保持字段结构)。百度优先,失败回退 LLM。"""
    if _baidu_available():
        try:
            return _translate_summary_baidu(summary, target, glossary)
        except Exception as exc:  # pragma: no cover - 网络/字段错位
            logger.warning("百度翻译摘要失败,回退 LLM: %s", exc)
    return llm.translate_summary(
        summary, target=target, glossary=glossary, model=model, api_key=api_key, base_url=base_url
    )


def translate_segments(
    segments: list[dict],
    *,
    target: str = "en",
    glossary: str = "",
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> list[str]:
    """逐段翻译,返回与输入等长的译文列表,用于双语对照展示。

    LLM 优先(一次调用翻完一批,质量与一致性更好);LLM 不可用时退回百度逐段翻译。
    """
    if not segments:
        return []
    try:
        return llm.translate_segments(
            segments,
            target=target,
            glossary=glossary,
            model=model,
            api_key=api_key,
            base_url=base_url,
        )
    except Exception as exc:  # pragma: no cover - 网络/配额相关
        logger.warning("LLM 逐段翻译失败,回退百度: %s", exc)
    if not _baidu_available():
        return ["" for _ in segments]
    out: list[str] = []
    for seg in segments:
        try:
            out.append(_baidu_translate_text(seg.get("text", ""), target, glossary))
        except Exception:  # pragma: no cover
            out.append("")
    return out


def _translate_summary_baidu(summary: dict, target: str, glossary: str = "") -> dict:
    """把纪要的所有非空字符串值按遍历顺序拼接成一行行,单次翻译后按原结构重建。"""
    texts: list[str] = []

    def collect(obj) -> None:
        if isinstance(obj, str):
            if obj.strip():
                texts.append(obj.strip())
        elif isinstance(obj, list):
            for x in obj:
                collect(x)
        elif isinstance(obj, dict):
            for v in obj.values():
                collect(v)

    collect(summary)
    if not texts:
        return summary

    joined = "\n".join(texts)
    protected, mapping = _protect(joined, _parse_glossary(glossary))
    parts = _restore(_baidu_request(protected, target), mapping).split("\n")
    if len(parts) != len(texts):
        raise RuntimeError(f"百度摘要字段数不一致({len(parts)} vs {len(texts)})")

    it = iter(parts)

    def rebuild(obj):
        if isinstance(obj, str):
            return next(it) if obj.strip() else obj
        if isinstance(obj, list):
            return [rebuild(x) for x in obj]
        if isinstance(obj, dict):
            return {k: rebuild(v) for k, v in obj.items()}
        return obj

    return rebuild(summary)
