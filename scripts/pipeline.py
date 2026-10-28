"""端到端管线:音频 → ASR 转写 → LLM 纠错 → LLM 摘要。

用法:
    # 从音频开始(先转写)
    python scripts/pipeline.py data/samples/测试1.m4a --glossary "DeepSeek"

    # 从已有转写 JSON 开始(跳过 ASR)
    python scripts/pipeline.py data/output/测试1.json --glossary "DeepSeek"

    # 从纯文本开始
    python scripts/pipeline.py data/output/测试1.txt --glossary "DeepSeek"

输出写入 data/output/<名>.summary.json,内含:
    transcript  原始转写(仅音频输入时)
    corrected   纠错后分段
    summary     结构化摘要
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import settings  # noqa: E402
from app.services.asr import (  # noqa: E402
    build_initial_prompt,
    save_transcript,
    transcribe_file,
)
from app.services.diarize import diarize  # noqa: E402
from app.services.llm import correct_transcript, summarize  # noqa: E402
from app.services.speaker import (  # noqa: E402
    assign_speakers,
    merge_turns,
    render_dialogue,
)
from app.services.text import (  # noqa: E402
    deep_to_simplified,
    segments_to_simplified,
    to_simplified,
)

AUDIO_SUFFIXES = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus", ".mp4", ".webm", ".aac"}


def load_segments(path: Path) -> list[dict] | None:
    """从 JSON 转写结果里读 segments;不是 JSON 返回 None。"""
    if path.suffix.lower() != ".json" or not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("segments")


def print_summary(summary: dict) -> None:
    print("\n" + "=" * 50)
    print(f"主题:{summary.get('title', '')}")
    print(f"概述:{summary.get('summary', '')}")
    topics = summary.get("topics", []) or []
    if topics:
        print("\n【议题】")
        for t in topics:
            print(f"  · {t.get('title', '')}")
            for p in t.get("points", []) or []:
                print(f"      - {p}")
            if t.get("conclusion"):
                print(f"      结论:{t['conclusion']}")
    actions = summary.get("action_items", []) or []
    if actions:
        print("\n【待办】")
        for a in actions:
            owner = a.get("owner") or "未定"
            deadline = a.get("deadline") or "未定"
            print(f"  · {a.get('task', '')}(负责人:{owner},截止:{deadline})")
    keywords = summary.get("keywords", []) or []
    if keywords:
        print(f"\n【关键词】{'、'.join(keywords)}")
    print("=" * 50)


def main() -> int:
    p = argparse.ArgumentParser(description="音频→纪要 端到端管线")
    p.add_argument("input", help="音频文件 / 转写 JSON / 纯文本")
    p.add_argument("--glossary", default="", help="术语表,逗号分隔")
    p.add_argument("--language", default=None, help="语言代码,如 zh")
    p.add_argument("--model", default=None, help="LLM 模型(默认读 .env)")
    p.add_argument("--skip-correct", action="store_true", help="跳过 LLM 纠错")
    p.add_argument("--no-diarize", action="store_true", help="关闭说话人分离")
    p.add_argument("--speakers", type=int, default=None, help="已知说话人数,可提升分离准确率")
    p.add_argument("--output-dir", default=None)
    args = p.parse_args()

    src = Path(args.input)
    if not src.exists():
        print(f"✗ 文件不存在: {src}", file=sys.stderr)
        return 1

    out_dir = Path(args.output_dir) if args.output_dir else settings.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) 拿到分段
    if src.suffix.lower() in AUDIO_SUFFIXES:
        cache = out_dir / f"{src.stem}.json"
        if cache.exists():
            print(f"▶ 复用已有转写 {cache.name} ...")
            data = json.loads(cache.read_text(encoding="utf-8"))
            segments = data["segments"]
            raw_text = data["text"]
        else:
            print(f"▶ 转写 {src.name} ...")
            prompt = build_initial_prompt(args.language, args.glossary)
            t = transcribe_file(src, language=args.language, initial_prompt=prompt)
            segments = [s for s in t.to_dict()["segments"]]
            raw_text = t.text
            save_transcript(t, cache)
            print(f"  → 转写缓存已存 {cache}")
    else:
        segments = load_segments(src)
        if segments is None:
            raw_text = src.read_text(encoding="utf-8")
            segments = [{"start": 0.0, "end": 0.0, "text": raw_text}]
        else:
            raw_text = "".join(s.get("text", "") for s in segments)

    # 1.5) 说话人分离(仅音频)
    if src.suffix.lower() in AUDIO_SUFFIXES and not args.no_diarize:
        if any(s.get("speaker") for s in segments):
            print("▶ 复用已有说话人标签 ...")
        else:
            print("▶ 识别不同讲话人 ...")
            spans = diarize(src, num_speakers=args.speakers)
            segments = assign_speakers(segments, spans)
            n_people = len({s["speaker"] for s in segments if s.get("speaker")})
            print(f"  → 识别到 {n_people} 位讲话人")

    # 2) 纠错
    if not args.skip_correct:
        print("▶ LLM 同音字纠错 + 术语统一 ...")
        corrected = segments_to_simplified(
            correct_transcript(segments, glossary=args.glossary, model=args.model)
        )
        corrected_text = to_simplified("".join(s.get("text", "") for s in corrected))
    else:
        corrected = segments_to_simplified(segments)
        corrected_text = to_simplified(raw_text)

    has_speaker = any(s.get("speaker") for s in corrected)
    dialogue = render_dialogue(corrected) if has_speaker else None

    # 3) 摘要(有说话人时按对话形式送入)
    print("▶ LLM 对话梳理 + 摘要 ...")
    summary = deep_to_simplified(
        summarize(dialogue or corrected_text, model=args.model)
    )

    result = {
        "source": str(src),
        "transcript": raw_text if src.suffix.lower() in AUDIO_SUFFIXES else None,
        "corrected_segments": corrected,
        "corrected_text": corrected_text,
        "dialogue": dialogue,
        "summary": summary,
    }

    out = out_dir / f"{src.stem}.summary.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / f"{src.stem}.summary.txt").write_text(
        dialogue or corrected_text, encoding="utf-8"
    )

    if not args.skip_correct:
        if dialogue:
            print("\n【对话记录(按讲话人)】")
            print("\n".join(dialogue.splitlines()[:20]))
            if len(dialogue.splitlines()) > 20:
                print(f"... 共 {len(dialogue.splitlines())} 轮发言")
        else:
            print("\n【纠错后文本】")
            print(corrected_text)
    print_summary(summary)
    print(f"\n✓ 已保存 → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
