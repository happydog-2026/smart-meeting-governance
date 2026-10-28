"""全局配置:统一从环境变量 / .env 读取。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path = ROOT / ".env") -> None:
    """极简 .env 加载器,避免为一个小功能引入 python-dotenv。

    已存在的环境变量优先,不会被 .env 覆盖。
    """
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


_load_dotenv()

# 国内网络默认走 HuggingFace 镜像,必须在导入 faster_whisper 之前设置。
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
# 新版 Hub 的 Xet 存储会绕过镜像并要求鉴权,直接关闭。
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
# Windows 不支持符号链接,关掉无意义的告警。
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# funasr 加载音频时用 subprocess 调系统 ffmpeg 二进制。为保持「全本地、免装
# 系统依赖」,把项目自带的 bin/ffmpeg.exe(pip 装 imageio-ffmpeg 提供)前置到 PATH。
# 必须在本模块(所有 app.services 的入口)导入期设置,早于 funasr 被 import。
_BIN = ROOT / "bin"
if _BIN.is_dir():
    os.environ["PATH"] = str(_BIN) + os.pathsep + os.environ.get("PATH", "")


@dataclass(frozen=True)
class Settings:
    # ASR
    whisper_model: str = os.getenv("WHISPER_MODEL", "small")
    # 默认 CPU:RTX 50 系(Blackwell)需 CUDA 12.8+,ctranslate2 4.8.4 在 Windows
    # 下不打包 cuBLAS 且不保证支持 sm_120,强行 GPU 会报 cublas64_12.dll 缺失。
    # 装好 CUDA Toolkit 12.8+ 后可设 WHISPER_DEVICE=cuda 尝试加速。
    whisper_device: str = os.getenv("WHISPER_DEVICE", "cpu")
    whisper_compute_type: str = os.getenv("WHISPER_COMPUTE_TYPE", "")
    whisper_model_dir: str = os.getenv("WHISPER_MODEL_DIR", "")

    # LLM(OpenAI 兼容接口,默认 DeepSeek)
    llm_api_key: str = os.getenv("LLM_API_KEY", "")
    llm_base_url: str = os.getenv("LLM_BASE_URL", "https://api.deepseek.com")
    llm_model: str = os.getenv("LLM_MODEL", "deepseek-chat")

    # 本地 Ollama(网页端 LLM 留空时的免费兜底,免 key、免联网、隐私好)
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")

    # 翻译(百度翻译 API,可选;留空则翻译退回 LLM)
    baidu_appid: str = os.getenv("BAIDU_TRANSLATE_APPID", "")
    baidu_key: str = os.getenv("BAIDU_TRANSLATE_KEY", "")

    # 中继后端 WebSocket 地址(前端组件据此连接;部署到远程机器时改成该机器的 IP/域名)
    rtasr_ws_url: str = os.getenv("RTASR_WS_URL", "ws://localhost:8000/rtasr")

    # 路径
    root: Path = ROOT
    data_dir: Path = ROOT / "data"
    output_dir: Path = ROOT / "data" / "output"


settings = Settings()
