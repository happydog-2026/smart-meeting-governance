"""只读分享页:/s/<token> 的 HTML 渲染。

页面完全静态(样式内联、不依赖 CDN / JS),把链接发到同办公室的电脑或手机上
也能正常打开;所有动态内容都经过 HTML 转义,会议文本里的尖括号不会破坏页面。
"""

from __future__ import annotations

import html

from app.services.speaker import format_time, merge_turns

#: 分享页上展示的品牌名(与 index.html 保持一致)
BRAND = "Smart Meeting Government"


def _e(value) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def _fmt_duration(seconds) -> str:
    try:
        total = int(round(float(seconds or 0)))
    except (TypeError, ValueError):
        return ""
    if total <= 0:
        return ""
    h, m, s = total // 3600, total % 3600 // 60, total % 60
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def _paragraphs(text: str) -> str:
    blocks = [ln.strip() for ln in str(text or "").splitlines() if ln.strip()]
    return "".join(f"<p>{_e(block)}</p>" for block in blocks)


def _topics_html(topics: list[dict]) -> str:
    cards: list[str] = []
    for topic in topics or []:
        title = _e(topic.get("title") or "未命名议题")
        points = "".join(f"<li>{_e(p)}</li>" for p in (topic.get("points") or []))
        points_html = f"<ul>{points}</ul>" if points else ""
        conclusion = topic.get("conclusion")
        conclusion_html = (
            f'<p class="conclusion"><span>结论</span>{_e(conclusion)}</p>'
            if conclusion
            else ""
        )
        cards.append(
            f'<section class="topic"><h3>{title}</h3>{points_html}{conclusion_html}</section>'
        )
    return "".join(cards)


def _actions_html(actions: list[dict]) -> str:
    items: list[str] = []
    for action in actions or []:
        task = _e(action.get("task") or "")
        tags = []
        if action.get("owner"):
            tags.append(f"负责人 {_e(action['owner'])}")
        if action.get("deadline"):
            tags.append(f"截止 {_e(action['deadline'])}")
        tag_html = f'<span class="tagitem">{" · ".join(tags)}</span>' if tags else ""
        items.append(f'<li><span class="box"></span><span class="task">{task}</span>{tag_html}</li>')
    return f'<ul class="actions">{"".join(items)}</ul>'


def _keywords_html(keywords: list) -> str:
    chips = "".join(f'<span class="chip">{_e(k)}</span>' for k in keywords or [] if str(k).strip())
    return f'<div class="chips">{chips}</div>' if chips else ""


def _transcript_html(result: dict) -> str:
    segments = result.get("corrected_segments") or []
    if segments:
        show_time = bool(result.get("has_timestamps", True))
        rows: list[str] = []
        for turn in merge_turns(segments, max_gap=3.0):
            who = f'<span class="who">{_e(turn.get("speaker") or "发言人")}</span>'
            ts = f'<span class="ts">[{format_time(turn.get("start") or 0)}]</span>' if show_time else ""
            rows.append(f"<p>{who}{ts}{_e(turn.get('text'))}</p>")
        if rows:
            return "".join(rows)
    text = (result.get("corrected_text") or result.get("transcript") or "").strip()
    return _paragraphs(text)


def _speaker_count(result: dict, meta: dict) -> int:
    if meta.get("n_speakers"):
        return int(meta["n_speakers"])
    return len(
        {
            seg.get("speaker")
            for seg in (result.get("corrected_segments") or [])
            if seg.get("speaker")
        }
    )


def render_share_page(result: dict, meta: dict | None = None) -> str:
    """把一次运行的完整结果渲染成只读分享页 HTML。"""
    meta = meta or {}
    summary = result.get("summary") or {}
    title = summary.get("title") or result.get("source") or meta.get("source_name") or "会议纪要"
    created = meta.get("created_at") or ""
    duration = _fmt_duration(result.get("duration") or meta.get("duration"))
    speakers = _speaker_count(result, meta)

    meta_bits = [created] if created else []
    if duration:
        meta_bits.append(f"时长 {duration}")
    if speakers:
        meta_bits.append(f"{speakers} 位发言人")
    meta_line = " · ".join(meta_bits)

    brief = (result.get("brief") or "").strip()
    brief_html = f'<div class="brief"><span>精简总结</span>{_paragraphs(brief)}</div>' if brief else ""

    summary_text = (summary.get("summary") or "").strip()
    summary_html = (
        f'<section class="block summary"><h2>会议概述</h2>{_paragraphs(summary_text)}</section>'
        if summary_text
        else ""
    )

    topics = summary.get("topics") or []
    topics_html = (
        f'<section class="block"><h2>议题与结论</h2>{_topics_html(topics)}</section>'
        if topics
        else ""
    )

    actions = summary.get("action_items") or []
    actions_html = (
        f'<section class="block"><h2>待办事项</h2>{_actions_html(actions)}</section>'
        if actions
        else ""
    )

    keywords_html = _keywords_html(summary.get("keywords") or [])
    keywords_block = (
        f'<section class="block"><h2>关键词</h2>{keywords_html}</section>' if keywords_html else ""
    )

    transcript = _transcript_html(result)
    transcript_html = (
        f'<details class="block transcript"><summary>原始转写</summary><div class="dialog">{transcript}</div></details>'
        if transcript
        else ""
    )

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<meta name="robots" content="noindex, nofollow" />
<title>{_e(title)} · 会议纪要</title>
<style>
  :root {{ color-scheme: light; }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; padding: 32px 16px 56px;
    background: #f6f7fb;
    color: #1e293b;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
    line-height: 1.75;
    -webkit-font-smoothing: antialiased;
  }}
  .wrap {{ max-width: 860px; margin: 0 auto; }}
  header.top {{
    display: flex; align-items: center; justify-content: space-between; gap: 12px;
    margin-bottom: 14px; padding: 0 4px;
  }}
  .brand {{ display: inline-flex; align-items: center; gap: 8px; font-size: 14px; font-weight: 700; color: #4f46e5; }}
  .brand i {{ width: 22px; height: 22px; border-radius: 7px; background: linear-gradient(135deg, #6366f1, #8b5cf6); display: inline-block; }}
  .badge {{
    font-size: 11px; color: #6366f1; background: #eef2ff;
    border: 1px solid #e0e7ff; border-radius: 999px; padding: 2px 10px; white-space: nowrap;
  }}
  main.card {{
    background: #fff; border-radius: 18px; padding: 36px 40px 32px;
    box-shadow: 0 1px 2px rgba(15, 23, 42, .05), 0 16px 40px -24px rgba(15, 23, 42, .25);
  }}
  h1 {{ margin: 0 0 10px; font-size: 26px; line-height: 1.35; letter-spacing: -.01em; }}
  .meta {{ margin: 0 0 22px; color: #94a3b8; font-size: 12.5px; }}
  h2 {{
    margin: 0 0 12px; font-size: 15px; color: #334155;
    display: flex; align-items: center; gap: 8px;
  }}
  h2::before {{ content: ""; width: 4px; height: 15px; border-radius: 2px; background: #6366f1; display: inline-block; }}
  .block {{ margin-top: 28px; padding-top: 22px; border-top: 1px solid #eef1f6; }}
  .block:first-of-type {{ border-top: 0; padding-top: 0; }}
  .brief {{
    background: #f5f6ff; border: 1px solid #e6e8ff; border-radius: 12px;
    padding: 14px 18px; margin: 0 0 6px; color: #3730a3;
  }}
  .brief span {{
    display: block; margin-bottom: 4px; font-size: 11px; font-weight: 700;
    letter-spacing: .08em; color: #6366f1;
  }}
  .brief p {{ margin: 4px 0; }}
  p {{ margin: 8px 0; }}
  .summary p {{ color: #334155; }}
  .topic {{ background: #fafbfd; border: 1px solid #eef1f6; border-radius: 12px; padding: 14px 18px; margin: 10px 0; }}
  .topic h3 {{ margin: 0 0 6px; font-size: 14.5px; color: #1e293b; }}
  .topic ul {{ margin: 4px 0 0; padding-left: 20px; color: #475569; }}
  .topic li {{ margin: 3px 0; }}
  .conclusion {{ margin: 10px 0 0; color: #0f766e; background: #ecfdf5; border-radius: 8px; padding: 7px 10px; font-size: 13.5px; }}
  .conclusion span {{ font-weight: 700; margin-right: 8px; color: #0d9488; }}
  ul.actions {{ list-style: none; margin: 0; padding: 0; }}
  ul.actions li {{
    display: flex; align-items: flex-start; gap: 10px; flex-wrap: wrap;
    padding: 9px 0; border-bottom: 1px dashed #eef1f6;
  }}
  ul.actions li:last-child {{ border-bottom: 0; }}
  .box {{ width: 14px; height: 14px; margin-top: 5px; border: 1.5px solid #a5b4fc; border-radius: 4px; flex: none; }}
  .task {{ flex: 1; min-width: 220px; color: #334155; }}
  .tagitem {{ font-size: 12px; color: #64748b; background: #f1f5f9; border-radius: 6px; padding: 1px 8px; }}
  .chips {{ display: flex; flex-wrap: wrap; gap: 8px; }}
  .chip {{ font-size: 12.5px; color: #4f46e5; background: #eef2ff; border: 1px solid #e0e7ff; border-radius: 999px; padding: 2px 12px; }}
  details.transcript summary {{
    cursor: pointer; font-size: 15px; font-weight: 600; color: #334155;
    display: flex; align-items: center; gap: 8px; list-style: none;
  }}
  details.transcript summary::-webkit-details-marker {{ display: none; }}
  details.transcript summary::before {{ content: ""; width: 4px; height: 15px; border-radius: 2px; background: #6366f1; display: inline-block; }}
  details.transcript[open] summary {{ margin-bottom: 12px; }}
  .dialog {{ max-height: 60vh; overflow-y: auto; background: #fafbfd; border: 1px solid #eef1f6; border-radius: 12px; padding: 14px 18px; }}
  .dialog p {{ margin: 6px 0; color: #475569; font-size: 14px; }}
  .who {{ display: inline-block; margin-right: 8px; padding: 1px 8px; border-radius: 6px; background: #eef2ff; color: #4f46e5; font-size: 12px; font-weight: 600; }}
  .ts {{ color: #94a3b8; font-size: 11.5px; margin-right: 8px; font-variant-numeric: tabular-nums; }}
  footer {{
    margin-top: 18px; text-align: center; font-size: 12px; color: #94a3b8;
  }}
  footer .dot {{ margin: 0 6px; color: #cbd5e1; }}
  @media (max-width: 640px) {{
    body {{ padding: 18px 10px 40px; }}
    main.card {{ padding: 24px 18px; border-radius: 14px; }}
    h1 {{ font-size: 21px; }}
  }}
  @media print {{
    body {{ background: #fff; padding: 0; }}
    main.card {{ box-shadow: none; padding: 0; }}
    header.top, footer {{ display: none; }}
    details.transcript {{ display: none; }}
  }}
</style>
</head>
<body>
<div class="wrap">
  <header class="top">
    <span class="brand"><i></i>{_e(BRAND)}</span>
    <span class="badge">只读分享</span>
  </header>
  <main class="card">
    <h1>{_e(title)}</h1>
    <p class="meta">{_e(meta_line)}</p>
    {brief_html}
    {summary_html}
    {topics_html}
    {actions_html}
    {keywords_block}
    {transcript_html}
  </main>
  <footer>本页面为只读分享，内容由分享者提供<span class="dot">·</span>{_e(BRAND)}</footer>
</div>
</body>
</html>
"""
