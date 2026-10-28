"""命令行转写工具。

用法:
    python scripts/transcribe.py data/meeting.mp3
    python scripts/transcribe.py data/meeting.mp3 --language zh --model small
    python scripts/transcribe.py data/*.m4a --glossary "飞书,豆包,DeepSeek"

结果默认写入 data/output/<文件名>.json 与 .txt。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Windows 控制台默认 GBK,统一转 UTF-8 避免中文/符号乱码
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import settings  # noqa: E402
from app.services.asr import (  # noqa: E402
    Segment,
    build_initial_prompt,
    format_timestamp,
    save_transcript,
    transcribe_file,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="faster-whisper 会议音频转写")
    p.add_argument("audio", nargs="+", help="一个或多个音频文件")
    p.add_argument("--model", default=None, help="模型大小,如 tiny/base/small/medium/large-v3")
    p.add_argument("--language", default=None, help="语言代码,如 zh / en;默认自动检测")
    p.add_argument("--device", default=None, help="cuda / cpu;默认读 .env")
    p.add_argument("--compute-type", default=None, help="float16 / int8 / int8_float16")
    p.add_argument("--glossary", default="", help="术语表,逗号分隔,用于纠正专有名词")
    p.add_argument("--no-vad", action="store_true", help="关闭 VAD 静音过滤")
    p.add_argument("--output-dir", default=None, help="输出目录,默认 data/output")
    p.add_argument("--quiet", action="store_true", help="不打印逐句进度")
    return p


def main() -> int:
    args = build_parser().parse_args()
    out_dir = Path(args.output_dir) if args.output_dir else settings.output_dir
    prompt = build_initial_prompt(args.language, args.glossary)
    total_elapsed = 0.0

    for raw in args.audio:
        audio = Path(raw)
        print(f"\n▶ 转写 {audio.name} ...", flush=True)

        def on_segment(seg: Segment) -> None:
            if not args.quiet:
                print(f"  [{format_timestamp(seg.start)} → {format_timestamp(seg.end)}] {seg.text}", flush=True)

        try:
            transcript = transcribe_file(
                audio,
                model_size=args.model,
                language=args.language,
                initial_prompt=prompt,
                device=args.device,
                compute_type=args.compute_type,
                vad=not args.no_vad,
                on_segment=on_segment,
            )
        except Exception as exc:
            print(f"✗ {audio.name} 转写失败: {exc}", file=sys.stderr)
            continue

        json_path = save_transcript(transcript, out_dir / f"{audio.stem}.json")
        (out_dir / f"{audio.stem}.txt").write_text(transcript.text, encoding="utf-8")
        total_elapsed += transcript.elapsed_seconds

        speed = transcript.duration / transcript.elapsed_seconds if transcript.elapsed_seconds else 0
        print(
            f"✓ 完成: {len(transcript.segments)} 段 / "
            f"语言 {transcript.language}({transcript.language_probability:.0%}) / "
            f"音频 {format_timestamp(transcript.duration)} / "
            f"耗时 {transcript.elapsed_seconds:.1f}s({speed:.1f}x 实时)"
        )
        print(f"  → {json_path}")
        print(f"  → {json_path.with_suffix('.txt')}")

    if len(args.audio) > 1:
        print(f"\n总计耗时 {total_elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
