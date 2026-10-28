"""说话人日志(Speaker Diarization)—— 识别"谁在什么时候说话"。

使用 FunASR cam++ 声纹聚类(funasr 集成式:VAD 切段 + cam++ 聚类,模型从 ModelScope 下载)。
本模块只产出时间段,不产出文字;文字由 faster-whisper 提供,二者按时间对齐。

输出统一格式:
    [{"start": 0.0, "end": 8.6, "speaker": "发言人1"}, ...]
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

from app.config import settings  # noqa: F401  # 触发 PATH 注入,让 funasr 找到 bin/ffmpeg
from app.services.speaker import normalize_speaker_labels

logger = logging.getLogger(__name__)

@lru_cache(maxsize=1)
def load_model():
    """加载说话人分离模型(funasr 集成式:VAD 切段 + cam++ 声纹聚类)。

    funasr 1.4.16 已移除独立的 cam++ 分离管线(iic/speech_campplus_speaker-diarization_common
    不再注册),改为 paraformer-zh(转写)+ fsmn-vad(切段)+ cam++(聚类)一次完成。
    这里只取 spk 标签与时间戳,文字由 faster-whisper 提供、按时间对齐后丢弃本模型的转写。
    """
    from funasr import AutoModel

    logger.info("加载说话人分离模型: paraformer-zh + fsmn-vad + cam++")
    return AutoModel(
        model="paraformer-zh",
        vad_model="fsmn-vad",
        spk_model="cam++",
        device="cpu",
        disable_update=True,
    )


def _looks_like_ms(values: list[float], duration: float | None) -> bool:
    """判断时间戳单位是毫秒还是秒。"""
    if not values:
        return False
    peak = max(values)
    if duration and duration > 0:
        # 音频时长已知:明显超过时长说明是毫秒
        return peak > duration * 1.5
    return peak > 600


def _parse_result(result, duration: float | None) -> list[dict]:
    """把 FunASR 的各种返回结构统一成 [{start, end, speaker}]。"""
    raw: list[dict] = []

    items = result if isinstance(result, list) else [result]
    for item in items:
        if not isinstance(item, dict):
            continue

        # 形式 A:{"key": "SPK0", "value": [[start, end], ...]}
        if "value" in item and isinstance(item["value"], list):
            spk = item.get("key") or item.get("spk") or "SPK"
            for pair in item["value"]:
                if isinstance(pair, (list, tuple)) and len(pair) >= 2:
                    raw.append({"start": float(pair[0]), "end": float(pair[1]), "speaker": str(spk)})

        # 形式 B:{"sentence_info": [{"start": ms, "end": ms, "spk": 0}, ...]}
        for sent in item.get("sentence_info", []) or []:
            spk = sent.get("spk", sent.get("speaker", "SPK"))
            raw.append({
                "start": float(sent.get("start", 0)),
                "end": float(sent.get("end", 0)),
                "speaker": f"SPK{spk}" if not str(spk).startswith("SPK") else str(spk),
            })

    if not raw:
        raise RuntimeError(f"无法解析说话人分离结果: {str(result)[:300]}")

    # 统一单位
    times = [v for r in raw for v in (r["start"], r["end"])]
    if _looks_like_ms(times, duration):
        for r in raw:
            r["start"] /= 1000.0
            r["end"] /= 1000.0

    # 过滤异常段
    raw = [r for r in raw if r["end"] > r["start"]]
    return sorted(raw, key=lambda x: x["start"])


def diarize(
    audio_path: str | Path,
    *,
    num_speakers: int | None = None,
    duration: float | None = None,
) -> list[dict]:
    """识别音频中的说话人时间段。

    Args:
        audio_path: 音频文件路径。
        num_speakers: 已知说话人数时传入,可提升准确率;None 表示自动判断。
        duration: 音频时长(秒),用于判断时间戳单位。
    Returns:
        [{"start": 0.0, "end": 8.6, "speaker": "发言人1"}, ...]
    """
    path = Path(audio_path)
    if not path.exists():
        raise FileNotFoundError(f"音频文件不存在: {path}")

    model = load_model()
    # 集成式 cam++ 自动聚类说话人数;num_speakers 暂不参与(保留参数以兼容调用方)。
    result = model.generate(input=str(path))
    spans = _parse_result(result, duration)
    labeled = normalize_speaker_labels(spans)
    logger.info("说话人分离完成: %d 段 / %d 人", len(labeled), len({s["speaker"] for s in labeled}))
    return labeled
