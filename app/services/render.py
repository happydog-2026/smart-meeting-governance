"""结果渲染 / 导出:把一次运行的 result dict 转成 Markdown / SRT / JSON。

Streamlit 版(app/web.py)里也有一份等价实现;这里是给新版 FastAPI 网页界面
复用的服务层版本,避免接口层直接依赖 Streamlit 模块(导入 web.py 会执行整个页面)。
"""

from __future__ import annotations

import json
from typing import Any

from app.services import langs
from app.services.speaker import format_time, render_dialogue, speaker_stats


def merge_turns_bilingual(segments: list[dict], *, max_gap: float = 3.0) -> list[dict]:
    """把相邻同一讲话人的片段合并成一轮发言,附带逐段译文。"""
    turns: list[dict] = []
    for seg in segments:
        text = (seg.get("text") or "").strip()
        translated = (seg.get("text_translated") or "").strip()
        speaker = seg.get("speaker") or ""
        if turns and turns[-1].get("speaker") == speaker and (
            float(seg.get("start") or 0) - float(turns[-1].get("end") or 0) <= max_gap
        ):
            turns[-1]["text"] += text
            turns[-1]["translated"] += translated
            turns[-1]["end"] = seg.get("end") or turns[-1]["end"]
        else:
            turns.append(
                {
                    "start": seg.get("start") or 0.0,
                    "end": seg.get("end") or 0.0,
                    "speaker": speaker,
                    "text": text,
                    "translated": translated,
                }
            )
    return turns


def bilingual_to_markdown(segments: list[dict], *, with_timestamp: bool = True) -> str:
    """把逐段译文导出为「原文 / 译文」交替的 Markdown。"""
    turns = merge_turns_bilingual(segments)
    lines = ["# 双语对照", ""]
    for t in turns:
        prefix = f"[{format_time(t['start'])}] " if (t.get("speaker") and with_timestamp) else ""
        speaker = f"{t['speaker']}:" if t.get("speaker") else ""
        lines.append(f"**{prefix}{speaker}** {t['text']}")
        if t.get("translated"):
            lines.append(f"> {t['translated']}")
        lines.append("")
    return "\n".join(lines)


def markdown_paragraphs(text: str) -> str:
    """每个非空行之间用空行分隔,保证 Markdown 里保留段落。"""
    blocks = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return "\n\n".join(blocks)


def to_markdown(
    summary: dict,
    transcript_text: str,
    *,
    is_dialogue: bool = False,
    lang: str = "zh",
    brief: str = "",
) -> str:
    """把结构化纪要渲染成 Markdown。lang="en" 时使用英文标题。"""
    L = {
        "zh": {"title": "会议纪要", "brief": "精简总结", "actions": "待办事项", "conclusion": "结论",
               "keywords": "关键词", "dialogue": "对话记录(按讲话人)", "transcript": "原始转录(校对后)",
               "owner": "负责人", "due": "截止", "join": "、"},
        "en": {"title": "Meeting Minutes", "brief": "Brief Summary", "actions": "Action Items", "conclusion": "Conclusion",
               "keywords": "Keywords", "dialogue": "Dialogue (by speaker)", "transcript": "Transcript (corrected)",
               "owner": "Owner", "due": "Due", "join": ", "},
    }[lang]
    lines = [f"# {summary.get('title') or L['title']}", ""]
    if brief:
        lines += [f"**{L['brief']}:** {brief}", ""]
    if summary.get("summary"):
        lines += [summary["summary"], ""]
    for topic in summary.get("topics", []) or []:
        lines.append(f"## {topic.get('title', '')}")
        for p in topic.get("points", []) or []:
            lines.append(f"- {p}")
        if topic.get("conclusion"):
            lines.append(f"\n**{L['conclusion']}:** {topic['conclusion']}\n")
    actions = summary.get("action_items", []) or []
    if actions:
        lines.append(f"## {L['actions']}")
        for a in actions:
            lines.append(
                f"- [ ] {a.get('task', '')}"
                f"{('(' + L['owner'] + ':' + a['owner'] + ')') if a.get('owner') else ''}"
                f"{('(' + L['due'] + ':' + a['deadline'] + ')') if a.get('deadline') else ''}"
            )
        lines.append("")
    if summary.get("keywords"):
        lines.append(f"**{L['keywords']}:** {L['join'].join(summary.get('keywords', []))}")
    heading = L["dialogue"] if is_dialogue else L["transcript"]
    lines += ["", "---", "", f"## {heading}", "", markdown_paragraphs(transcript_text)]
    return "\n".join(lines)


def _srt_timestamp(seconds: float) -> str:
    ms = int(round(max(0.0, float(seconds or 0)) * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def to_srt(result: dict) -> str:
    """按原始分段生成 SRT 字幕(无时间戳时整篇一段)。"""
    segments = result.get("source_segments") or result.get("corrected_segments") or []
    blocks: list[str] = []
    if segments:
        for i, seg in enumerate(segments, 1):
            start = seg.get("start") or 0.0
            end = seg.get("end") or start
            if end <= start:
                end = start + max(2.0, len(seg.get("text") or "") / 8)
            speaker = f"{seg['speaker']}:" if seg.get("speaker") else ""
            text = (seg.get("text") or "").strip()
            blocks.append(
                f"{i}\n{_srt_timestamp(start)} --> {_srt_timestamp(end)}\n{speaker} {text}\n"
            )
    else:
        text = (result.get("corrected_text") or result.get("transcript") or "").strip()
        if text:
            blocks.append(f"1\n{_srt_timestamp(0)} --> {_srt_timestamp(10)}\n{text}\n")
    return "\n".join(blocks)


def export_payload(result: dict) -> dict[str, Any]:
    """完整 JSON 导出(字段与 Streamlit 版保持一致)。"""
    summary = result.get("summary") or {}
    corrected = result.get("corrected_segments") or []
    has_speaker = any(s.get("speaker") for s in corrected)
    show_time = bool(result.get("has_timestamps", True))
    source_name = result.get("source") or "未命名"
    return {
        "source": source_name,
        "summary": summary,
        "brief": result.get("brief") or "",
        "transcript": result.get("transcript") or "",
        "corrected_text": result.get("corrected_text") or "",
        "corrected_segments": corrected,
        "dialogue": render_dialogue(corrected, with_timestamp=show_time) if has_speaker else None,
        "speaker_stats": speaker_stats(corrected) if has_speaker else None,
        "translated_summary": result.get("translated_summary"),
        "translated_text": result.get("translated_text"),
        "translated_lang": result.get("translated_lang") or "",
    }


def export_text(result: dict, fmt: str) -> tuple[str, str]:
    """返回 (正文, MIME)。fmt: md / bilingual / json / srt / txt / translated。"""
    summary = result.get("summary") or {}
    corrected = result.get("corrected_segments") or []
    has_speaker = any(s.get("speaker") for s in corrected)
    show_time = bool(result.get("has_timestamps", True))
    dialogue = (
        render_dialogue(corrected, with_timestamp=show_time)
        if has_speaker
        else (result.get("corrected_text") or result.get("transcript") or "")
    )
    brief = result.get("brief") or ""
    if fmt == "json":
        return json.dumps(export_payload(result), ensure_ascii=False, indent=2), "application/json"
    if fmt == "srt":
        return to_srt(result), "application/x-subrip"
    if fmt == "bilingual":
        return bilingual_to_markdown(corrected, with_timestamp=show_time), "text/markdown"
    if fmt == "txt":
        return dialogue, "text/plain"
    if fmt == "translated":
        trans_summary = result.get("translated_summary") or summary
        trans_lang = result.get("translated_lang") or ""
        trans_text = result.get("translated_text") or dialogue
        md = to_markdown(
            trans_summary,
            trans_text,
            is_dialogue=has_speaker,
            lang=langs.markdown_lang(trans_lang),
        )
        return md, "text/markdown"
    return (
        to_markdown(summary, dialogue, is_dialogue=has_speaker, brief=brief),
        "text/markdown",
    )


def qa_context(result: dict) -> str:
    """拼给会议问答用的资料:纪要 + 精简总结 + 原文。"""
    summary = result.get("summary") or {}
    brief = result.get("brief") or ""
    corrected = result.get("corrected_segments") or []
    corrected_text = result.get("corrected_text") or ""
    n_speakers = len({s.get("speaker") for s in corrected if s.get("speaker")})
    parts: list[str] = []
    if summary.get("title"):
        parts.append(f"会议标题:{summary['title']}")
    if summary.get("summary"):
        parts.append(f"会议概述:{summary['summary']}")
    topics = summary.get("topics") or []
    if topics:
        lines = []
        for t in topics:
            title = t.get("title") or "未命名议题"
            seg = f"- {title}"
            if t.get("points"):
                seg += ":" + ";".join(str(p) for p in t["points"])
            if t.get("conclusion"):
                seg += f"(结论:{t['conclusion']})"
            lines.append(seg)
        parts.append("议题与结论:\n" + "\n".join(lines))
    actions = summary.get("action_items") or []
    if actions:
        lines = []
        for a in actions:
            line = f"- {a.get('task', '')}"
            if a.get("owner"):
                line += f"(负责人:{a['owner']})"
            if a.get("deadline"):
                line += f"(截止:{a['deadline']})"
            lines.append(line)
        parts.append("待办事项:\n" + "\n".join(lines))
    if summary.get("keywords"):
        parts.append("关键词:" + "、".join(str(k) for k in summary["keywords"]))
    if brief:
        parts.append(f"精简总结:{brief}")
    transcript = (
        render_dialogue(corrected, with_timestamp=False) if n_speakers else corrected_text
    )
    parts.append("会议原文:\n" + transcript)
    return "\n\n".join(parts)
