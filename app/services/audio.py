"""音频基础处理:时长探测等轻量工具。

用项目自带的 bin/ffmpeg.exe(由 app.config 前置到 PATH),不依赖系统安装。
"""

from __future__ import annotations

import logging
import re
import subprocess
from pathlib import Path

from app.config import settings  # noqa: F401  # 触发 PATH 注入

logger = logging.getLogger(__name__)

_DURATION_RE = re.compile(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)")


def probe_duration(path: str | Path) -> float | None:
    """用 ffmpeg 读音频时长(秒);失败返回 None。

    并行流水线里,说话人分离需要时长来判断时间戳单位(秒/毫秒),
    而它要早于 ASR 完成,所以单独探测一次(开销约 0.1s)。
    """
    try:
        proc = subprocess.run(
            ["ffmpeg", "-hide_banner", "-i", str(path)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=20,
        )
    except Exception as exc:  # pragma: no cover - 环境相关
        logger.warning("ffmpeg 探测时长失败: %s", exc)
        return None

    # ffmpeg 读不到输出文件时以非零码退出,但 Duration 已打印在 stderr
    match = _DURATION_RE.search(proc.stderr or "")
    if not match:
        return None
    hours, minutes, seconds = match.groups()
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
