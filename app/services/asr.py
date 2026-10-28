"""语音转写服务(ASR)—— 基于 faster-whisper。

对外只暴露三个入口:
    transcribe_file(path, ...)   -> Transcript   转写一段音频
    save_transcript(t, path)                     落盘为 JSON
    Transcript.to_dict()                         与后端 / LLM 模块交接的 JSON

输出契约(全队共用,不要私自改字段):
{
  "model": "small",
  "language": "zh",
  "language_probability": 0.99,
  "duration": 123.4,
  "elapsed_seconds": 45.2,
  "text": "整段全文……",
  "segments": [
    {"start": 0.0, "end": 3.2, "speaker": null, "text": "第一句话。"}
  ]
}

说明:
- faster-whisper 自带 PyAV 解码,不依赖系统 ffmpeg,mp3/m4a/mp4 可直接读。
- 长音频通过内置 VAD 切片处理,不会一次性吃满显存。
- 说话人分离不在本模块职责内,`speaker` 字段先留空。
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Callable

from app.config import settings  # noqa: F401  必须先于 faster_whisper 导入,用于设置 HF_ENDPOINT

from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = {
    ".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus",
    ".mp4", ".webm", ".aac", ".wma", ".amr",
}

# 默认提示词:让 Whisper 输出带标准标点的简体中文(不传时中文常无标点)
ZH_PROMPT = "以下是普通话会议的录音转写,请使用简体中文和标准标点符号。"


def build_initial_prompt(language: str | None = None, glossary: str = "") -> str | None:
    """构造 ASR 提示词:中文标点约束 + 术语表。"""
    parts = []
    if language in (None, "zh"):
        parts.append(ZH_PROMPT)
    if glossary.strip():
        parts.append(f"会议中出现的专有名词包括:{glossary.strip()}。")
    return "".join(parts) or None


@dataclass
class Segment:
    start: float
    end: float
    text: str
    speaker: str | None = None


@dataclass
class Transcript:
    text: str
    segments: list[Segment] = field(default_factory=list)
    language: str = ""
    language_probability: float = 0.0
    duration: float = 0.0
    elapsed_seconds: float = 0.0
    model: str = ""

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "language": self.language,
            "language_probability": self.language_probability,
            "duration": self.duration,
            "elapsed_seconds": self.elapsed_seconds,
            "text": self.text,
            "segments": [asdict(s) for s in self.segments],
        }


def _detect_device() -> str:
    """有可用 CUDA 设备就上 GPU,否则 CPU。"""
    try:
        import ctranslate2

        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda"
    except Exception:  # pragma: no cover - 环境相关
        pass
    return "cpu"


@lru_cache(maxsize=4)
def load_model(
    model_size: str | None = None,
    device: str | None = None,
    compute_type: str | None = None,
) -> WhisperModel:
    """加载并缓存模型。同一组参数只会真正加载一次。"""
    size = model_size or settings.whisper_model
    dev = device or settings.whisper_device
    if dev == "auto":
        dev = _detect_device()
    ctype = compute_type or settings.whisper_compute_type or (
        "float16" if dev == "cuda" else "int8"
    )
    download_root = settings.whisper_model_dir or None

    try:
        model = WhisperModel(
            size,
            device=dev,
            compute_type=ctype,
            download_root=download_root,
            num_workers=1,
        )
        logger.info("ASR 模型已加载: %s / %s / %s", size, dev, ctype)
        return model
    except Exception as exc:  # CUDA/cuDNN 缺失时自动降级
        if dev == "cpu":
            raise
        logger.warning("GPU 加载失败(%s),回退 CPU int8", exc)
        return WhisperModel(
            size,
            device="cpu",
            compute_type="int8",
            download_root=download_root,
            num_workers=1,
        )


def transcribe_file(
    audio_path: str | Path,
    *,
    model_size: str | None = None,
    language: str | None = None,
    initial_prompt: str | None = None,
    device: str | None = None,
    compute_type: str | None = None,
    beam_size: int = 5,
    vad: bool = True,
    on_segment: Callable[[Segment], None] | None = None,
) -> Transcript:
    """转写一段音频。

    Args:
        audio_path: 音频/视频文件路径。
        model_size: 覆盖 .env 中的 WHISPER_MODEL。
        language: 指定语言(如 "zh"/"en");None 表示自动检测。
        initial_prompt: 提示词,可塞入术语表纠正专有名词。
        on_segment: 每产出一个片段回调一次,用于实时显示进度。
    """
    path = Path(audio_path)
    if not path.exists():
        raise FileNotFoundError(f"音频文件不存在: {path}")
    if path.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise ValueError(
            f"不支持的音频格式: {path.suffix},支持 {sorted(SUPPORTED_SUFFIXES)}"
        )

    model = load_model(model_size, device, compute_type)
    started = time.perf_counter()

    segments_iter, info = model.transcribe(
        str(path),
        language=language,
        initial_prompt=initial_prompt,
        beam_size=beam_size,
        vad_filter=vad,
        # 会议场景:说话间隙普遍较长,阈值放宽松一些
        vad_parameters={"min_silence_duration_ms": 500, "speech_pad_ms": 200},
        # 长音频下关闭上下文继承,避免同一句话被反复"幻觉"复读
        condition_on_previous_text=False,
        temperature=[0.0, 0.2, 0.4],
        no_speech_threshold=0.6,
    )

    segments: list[Segment] = []
    for s in segments_iter:
        text = s.text.strip()
        if not text:
            continue
        # 过滤 Whisper 在静音段常见的整段幻觉(高 no_speech 概率 + 低置信度)
        if s.no_speech_prob > 0.9 and s.avg_logprob < -1.0:
            logger.debug("丢弃疑似幻觉片段: %.1f-%.1f %r", s.start, s.end, text)
            continue
        seg = Segment(start=round(s.start, 2), end=round(s.end, 2), text=text)
        segments.append(seg)
        if on_segment:
            on_segment(seg)

    elapsed = time.perf_counter() - started
    transcript = Transcript(
        text="".join(seg.text for seg in segments).strip(),
        segments=segments,
        language=info.language or "",
        language_probability=round(info.language_probability or 0.0, 3),
        duration=round(info.duration or 0.0, 2),
        elapsed_seconds=round(elapsed, 2),
        model=model_size or settings.whisper_model,
    )
    logger.info(
        "转写完成: %.1fs 音频 / 耗时 %.1fs / %d 段 / 语言 %s",
        transcript.duration,
        transcript.elapsed_seconds,
        len(transcript.segments),
        transcript.language,
    )
    return transcript


def save_transcript(transcript: Transcript, out_path: str | Path) -> Path:
    """把转写结果写成 UTF-8 JSON。"""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(transcript.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return out


def format_timestamp(seconds: float) -> str:
    """秒 -> HH:MM:SS,用于命令行展示。"""
    total = int(seconds)
    return f"{total // 3600:02d}:{total % 3600 // 60:02d}:{total % 60:02d}"
