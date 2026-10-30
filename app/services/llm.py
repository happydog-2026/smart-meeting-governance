"""LLM 服务 —— 封装 DeepSeek(OpenAI 兼容接口)。

对外三个能力(对应比赛需求):
    correct_transcript()  同音字纠错 + 特殊术语统一
    summarize()           对话梳理 + 结构化摘要
    translate()           中英互译(预留)

说明:
- 所有函数输出都是 JSON(通过 response_format=json_object 强约束),失败自动重试。
- 术语表经 prompt 注入,用于纠正专有名词、统一译名。
"""

from __future__ import annotations

import json
import logging
import re
import threading
from functools import lru_cache
from typing import Any

import openai
from openai import OpenAI

from app.config import settings
from app.services import langs
from app.services.speaker import normalize_speaker_labels

logger = logging.getLogger(__name__)

CORRECT_SYSTEM = (
    "你是中文语音转写校对助手。你的任务是逐字校对,而不是改写或润色。\n"
    "需要修正的只有两类:1) 同音字/近音字误判(如「大午天」→「大雾天」);"
    "2) 专有名词被音译(如「DEEPSEAK」→「DeepSeek」)。\n"
    "铁律:原文的每个字、每个标点符号(逗号「,」句号「。」问号「?」感叹号「!」)都要原样保留,"
    "只能把确实写错的字替换掉,其余一律一字不改地复制。禁止增删标点、禁止合并或拆分句子。\n"
    "输出严格 JSON。"
)

SUMMARY_SYSTEM = (
    "你是对话/会议内容梳理助手。请通读对话,提炼出结构化摘要。\n"
    "要求:概括准确、信息不遗漏关键结论与待办;"
    "若对话中不存在某项(如无待办、无明确结论),对应字段给空数组或空字符串,不要编造。"
    "输出严格 JSON。"
)

TRANSLATE_SYSTEM = (
    "你是专业会议翻译助手,擅长把中文与英语、日语、韩语、法语、德语、"
    "西班牙语、葡萄牙语、意大利语、俄语、阿拉伯语等多语种互译。\n"
    "要求:忠实原文,不增删信息;专业术语按术语表统一;"
    "输出目标语言的纯翻译结果,不要加解释、不要保留原文。\n"
    "输出严格 JSON。"
)


@lru_cache(maxsize=8)
def _make_client(api_key: str, base_url: str) -> OpenAI:
    return OpenAI(api_key=api_key, base_url=base_url)


def get_client(*, api_key: str | None = None, base_url: str | None = None) -> OpenAI:
    """构建(并按 api_key/base_url 缓存)OpenAI 客户端。

    api_key=None 表示「用 .env 配置」(仅命令行脚本/开发者本地用)。
    前端(网页)永远传非空字符串:要么用户自己的 key,要么留空时由 web.py 兜底成
    本地 Ollama(占位 key "ollama")。空字符串在此视为配置缺失,直接报错。
    """
    if api_key is None:
        key = settings.llm_api_key
        if not key:
            raise RuntimeError("未配置 LLM API Key:请复制 .env.example 为 .env 并填入 LLM_API_KEY")
    else:
        key = api_key
        if not key:
            raise RuntimeError("未配置 LLM API Key:请在「⚙️ 设置」页「🔑 大模型」填入你自己的 key")
    url = base_url or settings.llm_base_url
    return _make_client(key, url)


def _ollama_ready() -> bool:
    """探测本机 Ollama 是否在跑、且已拉取配置的模型(断网时的本地兜底)。"""
    import urllib.request

    # settings.ollama_base_url 形如 http://localhost:11434/v1,根端点提供 /api/tags
    root = settings.ollama_base_url.rstrip("/").rsplit("/v1", 1)[0]
    try:
        with urllib.request.urlopen(root + "/api/tags", timeout=2) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return False
    names = {m.get("name", "").split(":")[0] for m in data.get("models", [])}
    return settings.ollama_model.split(":")[0] in names


# ---- 「本次已回退本地模型」的记录 ----
# 回退是发生在 _chat_json 内部的自动行为,调用方拿不到返回值;
# 用模块级集合记下,由前端在结果页显式提示用户,避免「静默降级」。
# 生成流程会用线程池并发调用,故加锁。
_fallback_lock = threading.Lock()
_fallback_notices: set[str] = set()


def _note_fallback(url: str) -> None:
    with _fallback_lock:
        _fallback_notices.add(url)


def pop_fallback_notices() -> list[str]:
    """取出并清空「已回退本地模型」记录(供网页端提示用户)。"""
    with _fallback_lock:
        out = sorted(_fallback_notices)
        _fallback_notices.clear()
    return out


def clear_fallback_notices() -> None:
    """清空记录(每次生成前调用,避免上一轮的提示残留到本轮)。"""
    with _fallback_lock:
        _fallback_notices.clear()


def _chat_json(
    system: str,
    user: str,
    *,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    temperature: float = 0.0,
    max_retries: int = 2,
) -> dict[str, Any]:
    """调用 LLM 并要求返回 JSON;解析失败自动重试。

    容错:若云端端点连不上(DNS 解析失败 / 断网 / 超时)且本机 Ollama 就绪,
    自动回退本地模型重试 —— 用户填了 key 但处于离线环境时,「全本地」依然成立。
    仅在「连接层」失败时回退;key 错误(401)等业务错误照常抛出,不掩盖问题。
    """
    # 当前生效的 (key, url, model),回退时会被替换
    cur_key, cur_url, cur_model = api_key, base_url, model or settings.llm_model
    last_err: Exception | None = None
    # 回退只做一次,且不计入 JSON 重试次数
    tried_fallback = False
    attempt = 0
    while attempt <= max_retries:
        client = get_client(api_key=cur_key, base_url=cur_url)
        try:
            resp = client.chat.completions.create(
                model=cur_model or settings.llm_model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=temperature,
                response_format={"type": "json_object"},
            )
        except openai.APIConnectionError as exc:
            url = cur_url or settings.llm_base_url
            if not tried_fallback and url != settings.ollama_base_url and _ollama_ready():
                tried_fallback = True
                _note_fallback(url)
                logger.warning(
                    "LLM 端点不可达(%s),回退本机 Ollama(%s):%s",
                    url,
                    settings.ollama_model,
                    exc,
                )
                cur_key, cur_url, cur_model = (
                    "ollama",
                    settings.ollama_base_url,
                    settings.ollama_model,
                )
                continue  # 不占用重试次数,直接用本地模型
            raise
        raw = resp.choices[0].message.content
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            last_err = exc
            logger.warning("LLM 输出非 JSON(第 %d 次):%s", attempt + 1, exc)
            attempt += 1
    raise RuntimeError(f"LLM 多次返回非 JSON: {last_err}")


def _correct_chunk(
    segments: list[dict],
    glossary_block: str,
    model: str | None,
    api_key: str | None,
    base_url: str | None,
) -> list[dict]:
    """对一小段分段做纠错(单次 LLM 调用,输入输出都受控)。"""
    user = (
        f"{glossary_block}\n"
        f"以下是带时间戳的语音转写分段(JSON):\n"
        f"{json.dumps(segments, ensure_ascii=False)}\n\n"
        f"请输出 JSON,格式为 {{\"segments\": [{{\"start\": 0.0, \"end\": 8.6, "
        f"\"text\": \"校对后文本\"}}]}},保持每段 start/end 不变,只改 text。"
    )
    data = _chat_json(
        CORRECT_SYSTEM, user, model=model, api_key=api_key, base_url=base_url
    )
    out = data.get("segments", [])
    if not isinstance(out, list):
        raise RuntimeError(f"纠错返回结构异常: {data}")
    if len(out) != len(segments):
        logger.warning("纠错段数不一致(%d vs %d),用原文补齐", len(out), len(segments))
    # 始终以原始分段为骨架合并,保证 speaker 等字段不丢失
    by_idx = {round(s.get("start", -1), 2): s for s in out}
    merged = []
    for s in segments:
        fixed = by_idx.get(round(s.get("start", -1), 2), s)
        merged.append({**s, "text": fixed.get("text", s.get("text", ""))})
    return merged


def correct_transcript(
    segments: list[dict],
    *,
    glossary: str = "",
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    chunk_chars: int = 2000,
) -> list[dict]:
    """同音字纠错 + 术语统一(长文本自动分块,避免超 LLM 输出上限)。

    Args:
        segments: ASR 输出的分段列表 [{start, end, text}, ...]。
        glossary: 术语表,逗号分隔的专有名词。
        model: LLM 模型名,空则用 .env 配置。
        api_key: 调用方自己的 key,空则用 .env 配置。
        base_url: 调用方自定义的 OpenAI 兼容接口地址,空则用 .env 配置。
        chunk_chars: 每块最大字符数,超过则切块逐次调用。
    Returns:
        与输入同结构的列表,start/end 不变,text 被校对。
    """
    glossary_block = ""
    if glossary.strip():
        glossary_block = f"\n专有名词术语表(出现时请严格使用以下写法):{glossary.strip()}\n"

    total_chars = sum(len(s.get("text", "")) for s in segments)
    if total_chars <= chunk_chars:
        return _correct_chunk(segments, glossary_block, model, api_key, base_url)

    # 按字符数切块,保证块边界落在完整分段上
    chunks: list[list[dict]] = []
    cur: list[dict] = []
    cur_len = 0
    for s in segments:
        cur.append(s)
        cur_len += len(s.get("text", ""))
        if cur_len >= chunk_chars:
            chunks.append(cur)
            cur, cur_len = [], 0
    if cur:
        chunks.append(cur)

    logger.info("长文本纠错分 %d 块处理", len(chunks))
    result: list[dict] = []
    for i, chunk in enumerate(chunks, 1):
        logger.info("纠错分块 %d/%d(%d 段)", i, len(chunks), len(chunk))
        result.extend(_correct_chunk(chunk, glossary_block, model, api_key, base_url))
    return result


def summarize(
    text: str,
    *,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> dict[str, Any]:
    """对话梳理 + 结构化摘要。

    Args:
        text: 对话全文(纠错后的纯文本)。
        model / api_key / base_url: 调用方自定义,空则用 .env 配置。
    Returns:
        {
          "title": str, "summary": str,
          "topics": [{"title", "points": [], "conclusion"}],
          "action_items": [{"task", "owner", "deadline"}],
          "keywords": [str]
        }
    """
    user = (
        f"对话内容:\n{text}\n\n"
        f"请输出 JSON,包含以下字段:\n"
        f'  "title": 一句话主题;\n'
        f'  "summary": 2-3 句总体概述;\n'
        f'  "topics": 讨论议题数组,每项含 title、points(要点数组)、conclusion(结论);\n'
        f'  "action_items": 待办数组,每项含 task、owner(负责人,可空)、deadline(截止,可空);\n'
        f'  "keywords": 关键词数组。\n'
        f"不存在某项时给空数组或空字符串。"
    )
    return _chat_json(SUMMARY_SYSTEM, user, model=model, api_key=api_key, base_url=base_url)


BRIEF_SYSTEM = (
    "你是会议纪要撰写助手。请把整场会议浓缩成一篇小篇幅总结,"
    "篇幅控制在 100~200 字,用一段话讲清:会议主题、主要结论、关键待办。"
    "只依据会议内容,不编造。输出严格 JSON。"
)


TERMS_SYSTEM = (
    "你是会议术语整理助手。请从给定资料中提取需要在语音转写里校正、"
    "并在中英翻译时保持译名一致的专有名词与术语:人名、机构名、产品名、"
    "技术术语、缩写等。\n"
    "要求:只提取资料里真实出现的词,不要臆造;"
    "剔除「会议」「问题」「方案」这类泛用词;"
    "一律使用资料中的规范写法(保留原始大小写);去重。输出严格 JSON。"
)


def extract_terms(
    text: str,
    *,
    limit: int = 60,
    max_chars: int = 12000,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> list[str]:
    """从资料正文里提取术语(供术语表一键导入)。

    截断到 max_chars,避免整份 PPT 撑爆上下文;结果做去重与清洗。
    """
    body = text[:max_chars]
    user = (
        f"资料内容:\n{body}\n\n"
        f'请输出 JSON:{{"terms": ["术语1", "术语2"]}},最多 {limit} 个。'
    )
    data = _chat_json(TERMS_SYSTEM, user, model=model, api_key=api_key, base_url=base_url)
    raw = data.get("terms")
    if not isinstance(raw, list):
        raise RuntimeError(f"术语提取返回结构异常: {data}")

    out: list[str] = []
    seen: set[str] = set()
    for item in raw:
        term = str(item).strip()
        # 过滤空串、过长的整句(多半是模型把句子当术语了)
        if not term or len(term) > 40 or term.lower() in seen:
            continue
        seen.add(term.lower())
        out.append(term)
        if len(out) >= limit:
            break
    return out


def brief_summary(
    text: str,
    *,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> str:
    """生成一篇小篇幅总结(区别于结构化纪要,给用户一段可直接引用的浓缩版)。"""
    user = (
        f"会议内容:\n{text}\n\n"
        f'请输出 JSON:{{"brief": "小篇幅总结"}}'
    )
    data = _chat_json(BRIEF_SYSTEM, user, model=model, api_key=api_key, base_url=base_url)
    return str(data.get("brief") or data.get("summary") or "").strip()


QA_SYSTEM = (
    "你是这场会议的智能助手,既能回答关于会议内容的问题,也能帮用户做进一步加工。\n"
    "1) 涉及会议本身的事实(谁说了什么、结论、待办、时间/数字等)必须以给定内容为准,"
    "不要编造;确实找不到时,直说「会议中未提及」,并可以顺带提示用户能问什么。\n"
    "2) 但用户要求总结、改写、翻译、整理成表格/清单/邮件/周报、提建议或补充常识时,"
    "就正常发挥,不必因为「会议里没写」而机械拒绝。\n"
    "3) 回答自然、简洁、有条理;可用 Markdown(列表、表格、加粗、小标题)。"
    "用户用中文问就中文答,用英文问就用英文答。\n"
    "输出严格 JSON。"
)


def ask_meeting(
    text: str,
    question: str,
    *,
    history: list[dict] | None = None,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    temperature: float = 0.3,
) -> str:
    """就会议内容问答(可带历史问答上下文,支持追问与按需生成总结/纪要)。

    温度略高于纪要类任务(0.3),让回答更自然、不那么生硬;
    归纳类任务仍以给定内容为准,不影响事实准确性。
    """
    history_block = ""
    if history:
        turns = "\n".join(
            f"问:{h.get('q', '')}\n答:{h.get('a', '')}" for h in history[-10:]
        )
        history_block = f"\n此前问答(用于理解追问的上下文,回答时保持前后一致):\n{turns}\n"
    user = (
        f"会议资料:\n{text}\n"
        f"{history_block}\n"
        f"用户问题:\n{question}\n\n"
        f'请输出 JSON:{{"answer": "你的回答"}}'
    )
    data = _chat_json(
        QA_SYSTEM,
        user,
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
    )
    return str(data.get("answer") or "").strip()


SPEAKER_ANNOTATE_SYSTEM = (
    "你是会议对话说话人标注助手。\n"
    "你会收到一段没有标注说话人的转写/对话文本。请把它切分成若干段发言,"
    "并给每段贴上统一标签「发言人1」「发言人2」…,按首次出现顺序编号。\n"
    "铁律:1) 不要改写、增删原文任何一个字,只负责切分与贴标签;"
    "2) 同一人连续发言尽量合并为一段;3) 若全文明显只有一个人在陈述/汇报,"
    "全部标为「发言人1」。\n"
    "输出严格 JSON。"
)


def annotate_speakers(
    text: str,
    *,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> list[dict]:
    """用 LLM 给没有声纹信息的文本标注说话人,返回 [{start, end, speaker, text}]。

    适用于「粘贴文本 / 实时字幕」这类没有音频、无法跑声纹分离的输入。
    start/end 用递增的伪时间戳填充(仅用于下游对齐,不参与展示)。
    """
    user = (
        f"待标注文本:\n{text}\n\n"
        f'请输出 JSON:{{"segments": [{{"speaker": "发言人1", "text": "…"}}, …]}}\n'
        f"要求:segments 的 text 按顺序拼接后必须与原文完全一致,一个字符都不能少。"
    )
    data = _chat_json(
        SPEAKER_ANNOTATE_SYSTEM, user, model=model, api_key=api_key, base_url=base_url
    )
    out = data.get("segments")
    if not isinstance(out, list):
        raise RuntimeError(f"说话人标注返回结构异常: {data}")
    segments: list[dict] = []
    for i, s in enumerate(out):
        if not isinstance(s, dict):
            continue
        t = (s.get("text") or "").strip()
        if not t:
            continue
        sp = (s.get("speaker") or "").strip() or "发言人1"
        segments.append({"start": float(i), "end": float(i) + 0.5, "speaker": sp, "text": t})
    if not segments:
        raise RuntimeError("说话人标注结果为空")
    return normalize_speaker_labels(segments)


def _split_text(text: str, max_chars: int = 1500) -> list[str]:
    """把长文本按句末标点切成若干块,每块尽量不超过 max_chars。"""
    text = text.strip()
    if len(text) <= max_chars:
        return [text] if text else []
    chunks: list[str] = []
    start, n = 0, len(text)
    while start < n:
        end = min(start + max_chars, n)
        if end < n:
            # 在窗口内找最后一个句末标点作为断点,避免截断句子
            for i in range(end, start + max_chars // 2, -1):
                if text[i - 1] in "。!?.;!?…\n":
                    end = i
                    break
        chunks.append(text[start:end].strip())
        start = end
    return chunks


_SENTENCE_ENDS = "。!?.;!?…"


def _paragraphize(text: str, target_chars: int = 220) -> str:
    """确保译文有自然段落:已有换行当段落边界,超长段落按「句末标点 + 长度」切段。

    兜底小模型(如 qwen2.5:3b)未必按指令在译文里插空行,这里在句末标点处、
    长度达到 target_chars 时断段,保证输出始终有段落、不会整篇挤成一段。
    """
    text = text.strip()
    if not text:
        return text
    out: list[str] = []
    # 先按任意换行拆开(空行 / 单换行都当段落边界)
    for para in re.split(r"\n+", text):
        para = para.strip()
        if not para:
            continue
        if len(para) <= target_chars:
            out.append(para)
            continue
        # 超长段落:逐字累积,在句末标点后、长度达标时断段
        buf = ""
        for ch in para:
            buf += ch
            if ch in _SENTENCE_ENDS and len(buf) >= target_chars:
                out.append(buf.strip())
                buf = ""
        if buf.strip():
            out.append(buf.strip())
    return "\n\n".join(out)


def translate(
    text: str,
    *,
    target: str = "en",
    glossary: str = "",
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> str:
    """多语种翻译(长文本自动分块)。target 为语言代码,见 app/services/langs.py。

    会保留原文段落结构:先按空行分段,段内超长再按句末标点切块;同时要求
    LLM 把译文按语义分成自然段落(空行分隔),避免整篇挤成一段。
    """
    direction = langs.prompt_name(target)
    glossary_block = f"\n术语表(严格按此翻译):{glossary.strip()}\n" if glossary.strip() else ""
    # 按空行先分成段落;没有空行的整段文本(如 ASR 转写)当作一个段落,交给 LLM 自己分
    paragraphs = [p for p in re.split(r"\n\s*\n", text.strip()) if p.strip()] or [text.strip()]
    translated_paras: list[str] = []
    for para in paragraphs:
        parts: list[str] = []
        for chunk in _split_text(para):
            user = (
                f"{glossary_block}\n"
                f"请将以下内容翻译成{direction}:\n{chunk}\n\n"
                "请把译文按语义分成自然段落,段落之间用一个空行分隔,不要输出成一整段。\n"
                '输出 JSON:{"translation": "译文"}'
            )
            data = _chat_json(TRANSLATE_SYSTEM, user, model=model, api_key=api_key, base_url=base_url)
            parts.append(data.get("translation", ""))
        # 同一段落被切块时用换行拼接(避免各块译文首尾相接),段落之间再补空行
        translated_paras.append("\n".join(parts))
    # _paragraphize 兜底:小模型不插空行时,按句末标点 + 长度强制分段
    return _paragraphize("\n\n".join(translated_paras))


def translate_summary(
    summary: dict[str, Any],
    *,
    target: str = "en",
    glossary: str = "",
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> dict[str, Any]:
    """翻译结构化纪要,保持 JSON 结构与字段名不变,只翻译字符串值。"""
    direction = langs.prompt_name(target)
    glossary_block = f"\n术语表(严格按此翻译):{glossary.strip()}\n" if glossary.strip() else ""
    user = (
        f"{glossary_block}\n"
        f"请将以下会议纪要 JSON 的所有字符串值翻译成{direction},"
        f"保持 JSON 结构和字段名完全不变:\n"
        f"{json.dumps(summary, ensure_ascii=False)}\n\n"
        f"输出 JSON,结构与输入一致,只翻译值。"
    )
    data = _chat_json(TRANSLATE_SYSTEM, user, model=model, api_key=api_key, base_url=base_url)
    if not isinstance(data, dict):
        return summary
    # 缺失字段用原文兜底,避免翻译丢失
    return {**summary, **data}


SEGMENT_TRANSLATE_SYSTEM = (
    "你是专业会议翻译助手,擅长中文与英语、日语、韩语、法语、德语、西班牙语、"
    "葡萄牙语、意大利语、俄语、阿拉伯语等多语种互译。\n"
    "你会收到按顺序编号的对话分段,请逐条翻译。\n"
    "铁律:译文条数必须与输入条数完全一致,顺序一一对应;"
    "每条只翻译该条内容,禁止合并、拆分、增删信息;专业术语按术语表统一。\n"
    "输出严格 JSON。"
)


def _translate_segments_chunk(
    texts: list[str],
    target: str,
    glossary_block: str,
    model: str | None,
    api_key: str | None,
    base_url: str | None,
) -> list[str]:
    """翻译一小批分段(单次 LLM 调用),返回等长译文列表。"""
    direction = langs.prompt_name(target)
    numbered = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(texts))
    user = (
        f"{glossary_block}\n"
        f"请把下列 {len(texts)} 条对话分段逐条翻译成{direction},"
        f"条数必须保持 {len(texts)} 条:\n{numbered}\n\n"
        "如果某条内容较长,把该条译文按语义分成自然段落(段落之间用一个空行分隔),"
        "每条译文仍只占数组里的一个元素。\n"
        f'输出 JSON:{{"translations": ["第1条译文", "第2条译文", ...]}}'
    )
    data = _chat_json(
        SEGMENT_TRANSLATE_SYSTEM,
        user,
        model=model,
        temperature=0.0,
        api_key=api_key,
        base_url=base_url,
    )
    out = data.get("translations", [])
    if not isinstance(out, list):
        raise RuntimeError(f"逐段翻译返回结构异常: {data}")
    if len(out) != len(texts):
        raise RuntimeError(f"逐段翻译条数不一致({len(out)} vs {len(texts)})")
    return [str(x) if x is not None else "" for x in out]


def translate_segments(
    segments: list[dict],
    *,
    target: str = "en",
    glossary: str = "",
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    chunk_chars: int = 1500,
) -> list[str]:
    """逐段翻译,返回与输入等长的译文列表(用于「原文 + 译文」双语对照展示)。

    长文本按字符数分块,每块一次 LLM 调用;某块条数对不上时该块降级为空串,
    保证返回值长度始终与 segments 一致,调用方可安全按下标对齐。
    """
    if not segments:
        return []
    texts = [(s.get("text") or "").strip() for s in segments]
    glossary_block = f"\n术语表(严格按此翻译):{glossary.strip()}\n" if glossary.strip() else ""

    # 按字符数切块,块边界落在完整分段上
    bounds: list[tuple[int, int]] = []
    start, cur = 0, 0
    for i, t in enumerate(texts):
        cur += len(t)
        if cur >= chunk_chars:
            bounds.append((start, i + 1))
            start, cur = i + 1, 0
    if start < len(texts):
        bounds.append((start, len(texts)))

    result: list[str] = [""] * len(texts)
    for bi, (lo, hi) in enumerate(bounds, 1):
        chunk = texts[lo:hi]
        try:
            result[lo:hi] = _translate_segments_chunk(
                chunk, target, glossary_block, model, api_key, base_url
            )
        except Exception as exc:  # pragma: no cover - 网络/结构异常
            logger.warning("逐段翻译分块 %d/%d 失败,降级为逐条翻译: %s", bi, len(bounds), exc)
            for i in range(lo, hi):
                if not chunk[i - lo]:
                    result[i] = ""
                    continue
                try:
                    one = _translate_segments_chunk(
                        [chunk[i - lo]], target, glossary_block, model, api_key, base_url
                    )
                    result[i] = one[0]
                except Exception:  # pragma: no cover
                    result[i] = ""
    return result
