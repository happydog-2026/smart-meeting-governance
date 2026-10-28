"""说话人标签对齐与对话格式化。

职责:
- assign_speakers(): 把说话人时间段(来自 diarize)对齐到 ASR 分段
- merge_turns():     把同一说话人的连续分段合并成一"轮"发言
- render_dialogue(): 渲染成「发言人N:内容」的对话文本
"""

from __future__ import annotations

from bisect import bisect_right


def format_time(seconds: float) -> str:
    """秒 -> MM:SS 或 HH:MM:SS。"""
    total = int(seconds or 0)
    h, m, s = total // 3600, total % 3600 // 60, total % 60
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def normalize_speaker_labels(spans: list[dict]) -> list[dict]:
    """把任意说话人标识(SPK0/0/None)统一成「发言人1」「发言人2」,按首次出现排序。"""
    order: dict[str, str] = {}
    out = []
    for sp in sorted(spans, key=lambda x: x["start"]):
        raw = str(sp.get("speaker", ""))
        if raw not in order:
            order[raw] = f"发言人{len(order) + 1}"
        out.append({**sp, "speaker": order[raw]})
    return out


def assign_speakers(
    segments: list[dict],
    spans: list[dict],
    *,
    min_overlap_ratio: float = 0.05,
) -> list[dict]:
    """给每个 ASR 分段打说话人标签。

    规则:取与该分段时间重叠最多的说话人;重叠过少或无重叠时沿用上一段说话人。

    Args:
        segments: [{start, end, text}, ...]
        spans:    [{start, end, speaker}, ...],来自 diarize,已排好序
    """
    if not spans:
        return [{**s, "speaker": None} for s in segments]

    spans = sorted(spans, key=lambda x: x["start"])
    starts = [sp["start"] for sp in spans]

    result: list[dict] = []
    last_speaker: str | None = None
    for seg in segments:
        s, e = float(seg.get("start", 0.0)), float(seg.get("end", 0.0))
        # 只检查可能与 [s, e] 相交的候选段
        idx = max(0, bisect_right(starts, e) - 1)
        best_speaker, best_overlap = None, 0.0
        j = idx
        while j >= 0 and spans[j]["end"] > s:
            overlap = min(e, spans[j]["end"]) - max(s, spans[j]["start"])
            if overlap > best_overlap:
                best_overlap, best_speaker = overlap, spans[j]["speaker"]
            if spans[j]["start"] <= s and j == idx:
                j -= 1
                continue
            j -= 1
            if j < 0 or spans[j]["end"] <= s:
                break
        # 向前再看一段,避免刚好落在边界另一侧
        if idx + 1 < len(spans) and spans[idx + 1]["start"] < e:
            k = idx + 1
            while k < len(spans) and spans[k]["start"] < e:
                overlap = min(e, spans[k]["end"]) - max(s, spans[k]["start"])
                if overlap > best_overlap:
                    best_overlap, best_speaker = overlap, spans[k]["speaker"]
                k += 1

        duration = max(e - s, 1e-6)
        if best_speaker is None or best_overlap / duration < min_overlap_ratio:
            best_speaker = last_speaker
        result.append({**seg, "speaker": best_speaker})
        if best_speaker:
            last_speaker = best_speaker
    return result


def merge_turns(
    segments: list[dict],
    *,
    max_gap: float = 3.0,
) -> list[dict]:
    """把同一说话人的连续分段合并成一"轮"发言。

    Args:
        max_gap: 同一说话人相邻分段间隔小于该秒数时合并。
    """
    turns: list[dict] = []
    for seg in segments:
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        speaker = seg.get("speaker")
        start, end = float(seg.get("start", 0.0)), float(seg.get("end", 0.0))
        if turns and turns[-1]["speaker"] == speaker and start - turns[-1]["end"] <= max_gap:
            turns[-1]["text"] += text
            turns[-1]["end"] = end
        else:
            turns.append({"speaker": speaker, "start": start, "end": end, "text": text})
    return turns


def render_dialogue(
    segments: list[dict],
    *,
    with_timestamp: bool = True,
    max_gap: float = 3.0,
) -> str:
    """渲染成对话文本。

    输出示例:
        [00:12] 发言人1:大家好,现在开始周会。
        [00:18] 发言人2:好的,我先汇报进度。
    """
    turns = merge_turns(segments, max_gap=max_gap)
    lines = []
    for t in turns:
        prefix = f"[{format_time(t['start'])}] " if with_timestamp else ""
        speaker = t.get("speaker") or "发言人"
        lines.append(f"{prefix}{speaker}:{t['text']}")
    return "\n".join(lines)


def speaker_stats(segments: list[dict]) -> list[dict]:
    """统计每位说话人的发言时长与轮次,用于前端展示。"""
    stats: dict[str, dict] = {}
    for seg in segments:
        name = seg.get("speaker") or "未识别"
        item = stats.setdefault(name, {"speaker": name, "duration": 0.0, "segments": 0})
        item["duration"] += max(float(seg.get("end", 0)) - float(seg.get("start", 0)), 0.0)
        item["segments"] += 1
    for item in stats.values():
        item["duration"] = round(item["duration"], 1)
    return sorted(stats.values(), key=lambda x: -x["duration"])
