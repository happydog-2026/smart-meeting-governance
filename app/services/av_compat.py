"""PyAV 版本兼容垫片。

背景:faster-whisper 解码音频时写死了 ``av.open(..., metadata_errors="ignore")``
(.venv/Lib/site-packages/faster_whisper/audio.py),而 PyAV 19 起该参数已被删除。
在装有 av 19+ 的机器上,任何音频导入都会在转写阶段抛:

    TypeError: open() got an unexpected keyword argument 'metadata_errors'

requirements.txt 已把 av 钉在 ``>=14,<19``,但用户机器上若早就装着 av 19
(旧 .venv、手动装过等),启动器的「依赖体检」只看能否 import,不会自动降级。
这里在运行时包一层 ``av.open``:遇到不支持该参数的版本就丢掉该参数重试,
其余行为原样保留,让旧环境也能直接跑。
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_FLAG = "_meetnote_metadata_errors_compat"


def patch_av_metadata_errors() -> None:
    """让 faster-whisper 在 PyAV 19+ 上也能正常解码。幂等,可重复调用。"""
    try:
        import av
    except Exception:  # pragma: no cover - av 缺失时 faster_whisper 自己也起不来
        return
    if getattr(av, _FLAG, False):
        return

    original_open = av.open

    def open_compat(*args, **kwargs):
        try:
            return original_open(*args, **kwargs)
        except TypeError as exc:
            # 只对「不认识 metadata_errors」这一种 TypeError 降级重试,避免掩盖真实错误
            if "metadata_errors" in kwargs and "metadata_errors" in str(exc):
                kwargs.pop("metadata_errors")
                logger.warning(
                    "当前 PyAV 版本不支持 metadata_errors,已自动兼容处理;"
                    '如需彻底对齐依赖,可执行:pip install "av<19"'
                )
                return original_open(*args, **kwargs)
            raise

    open_compat.__doc__ = getattr(original_open, "__doc__", None)
    av.open = open_compat
    setattr(av, _FLAG, True)
