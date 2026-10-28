"""用 Windows 自带语音合成生成测试音频(离线,无需联网)。

用法:
    python scripts/make_sample.py                      # 生成默认中文样例
    python scripts/make_sample.py --text "自定义文本" --out data/samples/a.wav
    python scripts/make_sample.py --voice "Microsoft Zira Desktop" --lang en
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

# Windows 控制台默认 GBK,统一转 UTF-8 避免中文/符号乱码
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent

DEFAULT_ZH = (
    "大家好,现在开始本周的项目周会。首先请产品组汇报一下当前进展。"
    "目前我们已经完成了需求梳理,下周会交付第一版原型。技术组这边,"
    "接口文档预计周三前完成,语音识别模块已经开始联调。最后提醒一下,"
    "所有同学请在周五下午之前提交本周的工作总结。"
)


def synthesize(text: str, out: Path, voice: str, rate: int = 0) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    # 单引号字符串内用两个单引号表示一个单引号,避免英文撇号破坏脚本
    safe_text = text.replace("'", "''")
    ps = (
        "Add-Type -AssemblyName System.Speech; "
        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        f"$s.SelectVoice('{voice}'); "
        f"$s.Rate = {rate}; "
        f"$s.SetOutputToWaveFile('{out.resolve()}'); "
        f"$s.Speak('{safe_text}'); "
        "$s.Dispose()"
    )
    subprocess.run(["powershell", "-NoProfile", "-Command", ps], check=True)
    print(f"✓ 已生成 {out} ({out.stat().st_size / 1024:.0f} KB)")


def main() -> int:
    p = argparse.ArgumentParser(description="用 Windows TTS 生成测试音频")
    p.add_argument("--text", default=DEFAULT_ZH)
    p.add_argument("--out", default=str(ROOT / "data" / "samples" / "sample_zh.wav"))
    p.add_argument("--voice", default="Microsoft Huihui Desktop")
    p.add_argument("--rate", type=int, default=0, help="语速,-10 到 10")
    args = p.parse_args()

    try:
        synthesize(args.text, Path(args.out), args.voice, args.rate)
    except subprocess.CalledProcessError as exc:
        print(f"✗ 合成失败: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
