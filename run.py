"""跨平台启动器:自动创建虚拟环境、体检并安装依赖、启动前端。

用法:
    Windows  :双击 start_web.bat
    Mac/Linux:bash 启动.sh  或  python3 run.py

设计目标:让「没配过环境」的协作者也能一键跑起来,并且能从「装了一半」的环境里自愈。
- 首次运行自动建 .venv
- 每次启动都真实导入探测依赖(不只看 .venv 是否存在),
  能抓到 sniffio / pydantic 这类「传递依赖缺失」导致的 ModuleNotFoundError
- pip 默认走清华镜像,失败回退官方源
- 出错时打印清晰原因,不闪退
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
IS_WIN = sys.platform == "win32"
VENV_DIR = ROOT / ".venv"
VENV_PY = VENV_DIR / ("Scripts/python.exe" if IS_WIN else "bin/python")
REQUIREMENTS = ROOT / "requirements.txt"

MIN_PY = (3, 9)
MAX_PY = (3, 14)  # 3.14+ 时 torch/funasr 可能还没有安装包

# 轻量依赖:真实 import 一次,能顺带查出传递依赖(sniffio/pydantic/httpx...)缺失
LIGHT_MODULES = [
    "streamlit",
    "openai",
    "faster_whisper",
    "av",
    "imageio_ffmpeg",
    "opencc",
    "kaldi_native_fbank",
    "fastapi",
    "uvicorn",
    "websockets",
]
# 重型依赖:import 要好几秒,只做存在性检查
HEAVY_MODULES = ["torch", "funasr", "modelscope"]

# 优先国内镜像(国内直连 PyPI 常半路超时,正是「依赖装一半」的元凶)
PIP_INDEXES = ["https://pypi.tuna.tsinghua.edu.cn/simple", None]


def _banner() -> None:
    print("=" * 52)
    print("  会议智治与跨语言沟通平台  启动器")
    print("=" * 52)


def _pause() -> None:
    try:
        input("\n按回车键退出...")
    except EOFError:
        pass


def _fail(msg: str) -> int:
    print(f"\n[错误] {msg}")
    _pause()
    return 1


def _check_deps(python: Path) -> list[str]:
    """在目标虚拟环境里体检依赖,返回缺失模块名(找不到则为 pip 包名)。"""
    code = (
        "import importlib.util, json\n"
        f"light = {LIGHT_MODULES!r}\n"
        f"heavy = {HEAVY_MODULES!r}\n"
        "missing = []\n"
        "for m in light:\n"
        "    try:\n"
        "        importlib.import_module(m)\n"
        "    except ModuleNotFoundError as e:\n"
        "        missing.append(e.name or m)\n"
        "    except Exception:\n"
        "        pass  # 其它异常说明包在,只是环境受限\n"
        "for m in heavy:\n"
        "    try:\n"
        "        if importlib.util.find_spec(m) is None:\n"
        "            missing.append(m)\n"
        "    except Exception:\n"
        "        missing.append(m)\n"
        "print(json.dumps(sorted(set(missing))))\n"
    )
    try:
        proc = subprocess.run(
            [str(python), "-c", code], capture_output=True, text=True, timeout=180
        )
    except Exception:
        return ["<体检超时>"]
    if proc.returncode != 0:
        return ["<体检失败>"]
    try:
        return json.loads((proc.stdout or "[]").strip().splitlines()[-1])
    except Exception:
        return []


def _pip(python: Path, target: list[str]) -> bool:
    """依次尝试镜像源安装,target 为 pip 参数(如包名列表或 -r requirements.txt)。"""
    base = [str(python), "-m", "pip", "install", "--disable-pip-version-check"] + target
    for index in PIP_INDEXES:
        cmd = base + (["-i", index] if index else [])
        print(f"   -> 数据源:{index or '官方 PyPI'}", flush=True)
        if subprocess.call(cmd) == 0:
            return True
        print("   -> 该源失败,换下一个源重试...", flush=True)
    return False


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    _banner()

    ver = sys.version_info[:2]
    if ver < MIN_PY:
        return _fail(f"当前 Python 版本过低({sys.version}),需要 Python 3.9 及以上。")
    if ver >= MAX_PY:
        print(
            f"\n[提示] 检测到 Python {sys.version_info.major}.{sys.version_info.minor},"
            " 部分依赖(如 torch)可能尚无安装包。"
        )
        print("       若安装失败,建议改装 Python 3.11~3.13 后重试。")

    # ---- 1. 确保虚拟环境存在 ----
    if not VENV_PY.exists():
        print(f"\n==> 首次运行:正在用 {sys.executable} 创建虚拟环境...")
        try:
            venv.EnvBuilder(with_pip=True).create(VENV_DIR)
        except Exception as exc:
            return _fail(f"创建虚拟环境失败:{exc}")
        print("==> 虚拟环境已创建。")

    # ---- 2. 依赖体检(每次启动都做,可修复装了一半的环境) ----
    print("\n==> 正在检查依赖...", flush=True)
    missing = _check_deps(VENV_PY)

    if missing:
        print(f"    缺少:{'、'.join(missing)}")
        print("    正在自动补装(需联网,请勿关闭窗口)...")
        # 先按名字精确补装(能直接补上 sniffio 这类传递依赖)
        ok = _pip(VENV_PY, missing)
        if not ok:
            # 再退回按 requirements 全量安装
            print("    精确补装失败,改为按 requirements.txt 全量安装...")
            ok = _pip(VENV_PY, ["-r", str(REQUIREMENTS)])
        if not ok:
            return _fail(
                "依赖安装失败。请检查网络;若多次失败,可能是 Python 版本过新,"
                "建议改装 Python 3.11~3.13。详见 README「常见问题」。"
            )
        still = _check_deps(VENV_PY)
        if still:
            return _fail(
                f"安装后仍缺少:{'、'.join(still)}。\n"
                "       请把这行报错截图反馈给作者。"
            )
        print("    ✓ 依赖已补齐。")
    else:
        print("    ✓ 依赖完整。")

    # ---- 3. 启动中继后端(实时转写 WebSocket)----
    rtasr_port = os.environ.get("RTASR_PORT", "8000")
    backend_proc: subprocess.Popen | None = None
    try:
        backend_proc = subprocess.Popen(
            [str(VENV_PY), "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", rtasr_port, "--log-level", "warning"],
            cwd=str(ROOT),
        )
        print(f"    ✓ 实时转写中继已启动(ws://127.0.0.1:{rtasr_port}/rtasr)")
    except Exception as exc:  # 后端起不来不阻断前端
        print(f"    [提示] 实时转写中继启动失败(可忽略,将回退 Web Speech):{exc}")

    # ---- 4. 启动前端 ----
    print("\n==> 正在启动,浏览器会自动打开 http://localhost:8501")
    print("    (保持本窗口开启;关闭本窗口即退出程序)\n")
    try:
        subprocess.call([str(VENV_PY), "-m", "streamlit", "run", str(ROOT / "app" / "web.py")])
    finally:
        if backend_proc and backend_proc.poll() is None:
            backend_proc.terminate()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        _pause()
