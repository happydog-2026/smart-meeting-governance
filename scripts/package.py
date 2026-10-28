"""把项目打包成干净的源码 zip,分享给协作者(自动剔除隐私 / 密钥 / 大文件)。

用法:
    python scripts/package.py

产出:
    dist/会议智治_源码_<日期>.zip

包含(统一放在顶层文件夹「会议智治/」下):
    源码(app/、scripts/)、requirements.txt、.env.example、CLAUDE.md、
    README.md(使用说明)、run.py + start_web.bat / 启动.sh(一键启动),
    bin/ffmpeg.exe(说话人分离所需)、小体积样例音频。

不含(防止泄露):
    .env(真实 API Key)、data/llm_settings.json(记住的 LLM key)、
    data/xfyun_settings.json(记住的讯飞 key)、
    data/history.db 与 data/history/(历史会议数据)、data/output(个人结果)、
    .venv、models、__pycache__、大体积个人录音(余华测试2 / 英文测试1)。
"""

from __future__ import annotations

import os
import sys
from datetime import date
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
DIST = ROOT / "dist"

# 不打包的目录(相对路径,匹配到整个目录都跳过)
EXCLUDE_DIRS = {
    ".venv",
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".serena",
    "models",
    "dist",
    "data/output",
    "data/history",
}

# 不打包的具体文件(含密钥 / 隐私 / 个人结果)
EXCLUDE_FILES = {
    ".env",
    "data/llm_settings.json",  # 你「记住」的 LLM API Key
    "data/xfyun_settings.json",  # 你「记住」的讯飞 key
    "data/history.db",          # 历史会议数据
}

# 不打包的大体积个人录音(协作者可用小样例自测)
EXCLUDE_NAMES = {
    "余华测试2.m4a",
    "英文测试1.m4a",
}


def _rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def _excluded(rel: str) -> bool:
    parts = rel.split("/")
    for d in EXCLUDE_DIRS:
        if rel == d or rel.startswith(d + "/"):
            return True
        # 单段目录名(如 __pycache__)匹配任意嵌套层级
        if "/" not in d and d in parts:
            return True
    if rel in EXCLUDE_FILES:
        return True
    if parts[-1] in EXCLUDE_NAMES:
        return True
    # 编译缓存文件一并剔除
    if rel.endswith((".pyc", ".pyo")):
        return True
    return False


def main() -> int:
    DIST.mkdir(exist_ok=True)
    out = DIST / f"会议智治_源码_{date.today().strftime('%Y%m%d')}.zip"

    included: list[str] = []
    excluded: list[str] = []
    for dirpath, dirnames, filenames in os.walk(ROOT):
        # 剪枝:不进要排除的目录(避免遍历 2G 的 .venv)
        dirnames[:] = [d for d in dirnames if not _excluded(_rel(Path(dirpath) / d))]
        for fn in filenames:
            rel = _rel(Path(dirpath) / fn)
            (excluded if _excluded(rel) else included).append(rel)

    included.sort()
    with ZipFile(out, "w", ZIP_DEFLATED) as zf:
        for rel in included:
            # 统一放进顶层文件夹,避免解压时文件散落一地
            zf.write(ROOT / rel, f"会议智治/{rel}")

    # 复核:确认密钥/隐私文件确实没进包
    with ZipFile(out, "r") as zf:
        names = set(zf.namelist())
    leaked = [f for f in EXCLUDE_FILES if f in names]
    if leaked:
        print("❌ 打包异常:以下文件不应出现在包里:", leaked, file=sys.stderr)
        return 1

    size_mb = out.stat().st_size / 1024 / 1024
    print(f"✅ 打包完成 → {out}")
    print(f"   共 {len(included)} 个文件,{size_mb:.1f} MB")
    print("\n已剔除(隐私/密钥/大文件):")
    for rel in excluded:
        print(f"   ✗ {rel}")
    print("\n包内顶层结构:")
    for rel in included:
        if rel.count("/") <= 1:
            print(f"   · {rel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
