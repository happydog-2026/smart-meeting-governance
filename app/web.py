"""会议智治与跨语言沟通平台 —— Streamlit 前端。

启动:
    streamlit run app/web.py

功能:上传音频 / 粘贴文本 → ASR 转写 → LLM 纠错 → 结构化纪要,支持导出。
"""

from __future__ import annotations

import hashlib
import html as _html
import json
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import settings  # noqa: E402
from app.services.asr import build_initial_prompt, transcribe_file  # noqa: E402
from app.services.audio import probe_duration  # noqa: E402
from app.services.diarize import diarize  # noqa: E402
from app.services.llm import (  # noqa: E402
    annotate_speakers,
    ask_meeting,
    brief_summary,
    clear_fallback_notices,
    correct_transcript,
    extract_terms,
    pop_fallback_notices,
    summarize,
)
from app.services.extract import extract_text  # noqa: E402
from app.services import langs  # noqa: E402
from app.services.translate import translate, translate_segments, translate_summary  # noqa: E402
from app.services.speaker import (  # noqa: E402
    assign_speakers,
    format_time,
    merge_turns,
    render_dialogue,
    speaker_stats,
)
from app.services.text import (  # noqa: E402
    deep_to_simplified,
    segments_to_simplified,
    to_simplified,
)
from app.services.history import delete_run, get_run, list_runs, save_run  # noqa: E402


SPEAKER_COLORS = ["#2563eb", "#db2777", "#059669", "#d97706", "#7c3aed", "#0891b2"]

# 侧边栏「双语翻译」选项 → 语言代码。语种清单集中在 app/services/langs.py,
# 要加语种只改那里,这里和结果页的「目标语言」会自动跟着变。
TRANSLATE_TARGETS = {
    f"翻译成{v['label']}": code for code, v in langs.LANGUAGES.items()
}

# 处理流水线的四个步骤(顺序即展示顺序)
_PIPELINE_STEPS = [
    ("asr", "🎙️", "语音转写"),
    ("diar", "👥", "说话人识别"),
    ("correct", "✍️", "同音字纠错"),
    ("llm", "🧠", "纪要 + 翻译"),
]

_CSS = """
<style>
/* 步骤流水线卡片 */
.mc-pipeline { display:flex; gap:10px; flex-wrap:wrap; margin:4px 0 14px; }
/* 中性色一律用半透明灰:亮 / 暗两套主题下都能读,不用关心当前是哪套 */
.mc-step { flex:1 1 160px; min-width:140px; border-radius:12px; padding:12px 14px;
           background:rgba(127,127,127,.10); border:1px solid rgba(127,127,127,.28);
           transition:all .2s ease; }
.mc-step.running { background:rgba(59,130,246,.14); border-color:rgba(59,130,246,.55);
                   box-shadow:0 0 0 3px rgba(59,130,246,.14); }
.mc-step.done { background:rgba(34,197,94,.14); border-color:rgba(34,197,94,.5); }
.mc-step.failed { background:rgba(239,68,68,.14); border-color:rgba(239,68,68,.5); }
.mc-step .mc-icon { font-size:15px; }
.mc-step .mc-name { font-size:13px; font-weight:600; color:inherit; margin-top:2px; }
.mc-step .mc-detail { font-size:11.5px; color:inherit; opacity:.72; margin-top:3px; line-height:1.5; }
.mc-step.running .mc-detail { color:#3b82f6; opacity:1; }
/* 实时转写卡片 */
.mc-live { border:1px solid rgba(127,127,127,.28); border-radius:12px; padding:12px 14px;
           background:rgba(127,127,127,.06); max-height:300px; overflow-y:auto; }
.mc-live-head { font-size:12px; margin-bottom:8px; display:flex; gap:8px; align-items:center; }
.mc-live-row { display:flex; gap:10px; margin:4px 0; }
.mc-live-time { flex:0 0 48px; color:inherit; opacity:.55; font-size:12px; font-variant-numeric:tabular-nums; }
.mc-live-text { flex:1; line-height:1.65; color:inherit; }
/* 实时转写:灵动动画(呼吸圆点 / 最新一句滑入 / 打字光标) */
.mc-live-dot { display:inline-block; width:8px; height:8px; border-radius:50%;
               background:#22c55e; margin-right:6px; vertical-align:middle;
               animation:mcPulse 1.2s ease-in-out infinite; }
@keyframes mcPulse { 0%,100%{opacity:1;transform:scale(1);} 50%{opacity:.35;transform:scale(.65);} }
.mc-live-row.latest { background:linear-gradient(90deg, rgba(59,130,246,.07), transparent);
                      border-radius:8px; margin-left:-6px; padding-left:6px;
                      animation:mcFadeUp .35s ease both; }
@keyframes mcFadeUp { from{opacity:0;transform:translateY(6px);} to{opacity:1;transform:translateY(0);} }
.mc-live-cursor { display:inline-block; width:2px; height:1.05em; background:#2563eb;
                  vertical-align:text-bottom; margin-left:2px; animation:mcBlink 1s step-end infinite; }
@keyframes mcBlink { 0%,100%{opacity:1;} 50%{opacity:0;} }
/* 精简总结 */
.mc-brief { border-left:4px solid #2563eb; background:rgba(127,127,127,.10); border-radius:8px;
            padding:12px 16px; line-height:1.7; color:inherit; margin:6px 0 4px; }
</style>
"""


def _inject_css() -> None:
    """注入全局样式(自定义组件美化)。配色跟随 Streamlit 当前主题。"""
    st.markdown(_CSS, unsafe_allow_html=True)


def _pipeline_html(states: dict[str, tuple[str, str]]) -> str:
    """渲染四步流水线卡片。states: {key: (状态, 详情)}。

    状态取值:idle / running / done / failed
    """
    cards = []
    for key, icon, name in _PIPELINE_STEPS:
        state, detail = states.get(key, ("idle", ""))
        mark = {"done": "✅", "failed": "⚠️", "running": "⏳"}.get(state, "")
        cards.append(
            f'<div class="mc-step {state}">'
            f'<div class="mc-icon">{icon} <span class="mc-name">{name}</span></div>'
            f'<div class="mc-detail">{mark} {detail or "等待中…"}</div>'
            "</div>"
        )
    return f'<div class="mc-pipeline">{"".join(cards)}</div>'

# LLM 服务商 / 模型预设:用户可在侧边栏选择,key 会记住到本地文件
_LLM_PRESETS = {
    "DeepSeek · deepseek-chat": ("https://api.deepseek.com", "deepseek-chat"),
    "DeepSeek · deepseek-reasoner": ("https://api.deepseek.com", "deepseek-reasoner"),
    "OpenAI · gpt-4o-mini": ("https://api.openai.com/v1", "gpt-4o-mini"),
    "OpenAI · gpt-4o": ("https://api.openai.com/v1", "gpt-4o"),
    "通义千问 · qwen-plus": ("https://dashscope.aliyuncs.com/compatible-mode/v1", "qwen-plus"),
    "通义千问 · qwen-turbo": ("https://dashscope.aliyuncs.com/compatible-mode/v1", "qwen-turbo"),
}
_LLM_OPTIONS = list(_LLM_PRESETS.keys()) + ["自定义…"]
_LLM_SETTINGS_PATH = ROOT / "data" / "llm_settings.json"


def _load_llm_settings() -> dict:
    """读取上次保存的 LLM 配置(记住用户填过的 API Key / 模型选择)。"""
    try:
        return json.loads(_LLM_SETTINGS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_llm_settings(data: dict) -> None:
    """把 LLM 配置写入本地文件,下次启动自动恢复。"""
    _LLM_SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    _LLM_SETTINGS_PATH.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )


_XFYUN_SETTINGS_PATH = ROOT / "data" / "xfyun_settings.json"


def _load_xfyun_settings() -> dict:
    """读取上次保存的讯飞 key(用户自己的,用于覆盖 .env)。"""
    try:
        return json.loads(_XFYUN_SETTINGS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_xfyun_settings(data: dict) -> None:
    """把用户自己的讯飞 key 写入本地文件,下次启动自动恢复。"""
    _XFYUN_SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    _XFYUN_SETTINGS_PATH.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _ollama_available() -> bool:
    """探测本地 Ollama 是否在运行、且已拉取配置的模型(LLM 留空时的免费兜底)。"""
    import urllib.request

    # settings.ollama_base_url 形如 http://localhost:11434/v1,根端点提供 /api/tags
    root = settings.ollama_base_url.rstrip("/").rsplit("/v1", 1)[0]
    try:
        with urllib.request.urlopen(root + "/api/tags", timeout=2) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return False
    names = {m.get("name", "").split(":")[0] for m in data.get("models", [])}
    return settings.ollama_model.split(":")[0] in names


# ============ 主题 ============
# 明暗主题交给 Streamlit 自己管(右上角「⋮」→ 设置 → 主题;默认跟随系统)。
# 曾经这里用一份手写的 CSS 变量覆盖(:root{--background-color...})来做亮/暗切换,
# 但 Streamlit 1.66 的主题是编译进 emotion 样式的、并不消费这些变量:
# 结果是「容器背景被强制刷白、文字还是原主题的浅色」,暗色下白底白字糊成一片。
# 现在不再覆盖任何基础配色,自定义组件(.mc-*)也一律用半透明中性色,两套主题通用。


def _llm_runtime_config() -> tuple[str, str, str, bool]:
    """从 session_state 推导本次运行真正要用的 LLM 配置。

    返回 (api_key, base_url, model, using_ollama)。
    设置页只负责把用户输入写进 session_state;这里统一算出参数,保证主界面
    无论当前停留在哪个导航页,都能拿到一致、可用的 LLM 配置。
    """
    key = st.session_state.get("llm_api_key", "") or ""
    provider = st.session_state.get("llm_provider", "DeepSeek · deepseek-chat")
    if provider == "自定义…":
        base_url = st.session_state.get("llm_custom_url", "") or settings.llm_base_url
        model = st.session_state.get("llm_custom_model", "") or settings.llm_model
    else:
        base_url, model = _LLM_PRESETS.get(
            provider, _LLM_PRESETS["DeepSeek · deepseek-chat"]
        )
    using_ollama = not key
    if using_ollama:
        base_url, model, key = settings.ollama_base_url, settings.ollama_model, "ollama"
    return key, base_url, model, using_ollama


def _open_run(run_id: str) -> None:
    """历史记录「查看」按钮的回调:切到主界面并打开该条记录。

    必须放在按钮回调里,不能写在 if button 分支内:侧边栏 radio 的 key 就是 nav,
    控件实例化后再改 st.session_state["nav"] 会抛 StreamlitWidgetAlreadyInstantiatedError。
    回调在控件实例化之前执行,因此这里改 nav 是安全的。
    """
    st.session_state["view_run_id"] = run_id
    st.session_state["nav"] = "📝 主界面"


def _render_settings() -> None:
    """渲染「设置」页:主题说明 + 两个 API Key + 历史记录。"""
    st.subheader("🎨 主题")
    st.caption(
        "明暗主题用右上角「⋮」→ 设置 → 主题切换(默认跟随系统);"
        "界面配色全部跟随 Streamlit 主题,亮色 / 暗色下都能正常显示。"
    )

    st.divider()
    st.subheader("🔑 大模型 (LLM)")
    llm_provider = st.selectbox(
        "服务商 / 模型",
        _LLM_OPTIONS,
        key="llm_provider",
        help="选择要调用的大模型；选「自定义…」可填任意 OpenAI 兼容接口",
    )
    api_key = st.text_input(
        "API Key",
        type="password",
        key="llm_api_key_widget",
        placeholder="留空则用本地 Ollama(免费)",
        help="填入你自己的 LLM key 走云端;留空则回退本地 Ollama 免费模型(免 key、免联网)。key 会自动记住,不会用服务器 .env 里的 key",
    )
    # 密码框在 rerun 时会被 Streamlit 清空,手动存进 session_state 以便跨页面复用
    if api_key:
        st.session_state["llm_api_key"] = api_key
    saved_key = st.session_state.get("llm_api_key", "")

    if llm_provider == "自定义…":
        llm_base_url = st.text_input(
            "API Base URL", key="llm_custom_url", help="OpenAI 兼容接口地址"
        )
        llm_model = st.text_input("模型名称", key="llm_custom_model")
    else:
        llm_base_url, llm_model = _LLM_PRESETS[llm_provider]

    if saved_key:
        cfg = {
            "api_key": saved_key,
            "provider": llm_provider,
            "base_url": llm_base_url,
            "model": llm_model,
        }
        if cfg != _load_llm_settings():
            _save_llm_settings(cfg)
        tail = saved_key[-4:] if len(saved_key) >= 4 else ""
        st.caption(f"✅ 已记住 API Key(…{tail}),下次打开自动恢复")
        st.caption(
            "📴 断网或云端连不上时会自动回退本机 Ollama(结果页会明确提示);"
            "想强制走本地,点下面的「忘记已保存的 Key」即可。"
        )
        if st.button("🗑️ 忘记已保存的 Key", key="forget_llm_key"):
            _LLM_SETTINGS_PATH.unlink(missing_ok=True)
            st.session_state["llm_api_key"] = ""
            st.rerun()
    else:
        st.caption(
            f"🟢 未填 key:改用本地 Ollama({settings.ollama_model},免费、免联网)。"
            "若还没装,请先安装并启动 Ollama(ollama.com)。"
        )

    st.divider()
    st.subheader("🔊 实时转写 (讯飞)")
    st.caption(
        "⚠️ 实时字幕是云端功能:讯飞或浏览器 Web Speech 均需联网,不参与全本地离线。"
        "要离线转写,请回主界面用「🎤 本地录音」(本机 faster-whisper,免联网)。"
    )
    xfyun_appid = st.text_input(
        "AppID",
        type="password",
        key="xfyun_appid_widget",
        placeholder="留空则不启用云端,回退 Web Speech API",
        help="填入你自己的讯飞 AppID 即启用云端实时转写;会自动记住,下次打开无需重填",
    )
    if xfyun_appid:
        st.session_state["xfyun_appid"] = xfyun_appid
    xfyun_apikey = st.text_input(
        "接口密钥 (APIKey)",
        type="password",
        key="xfyun_apikey_widget",
        placeholder="留空则不启用云端,回退 Web Speech API",
        help="「实时语音转写」服务的接口密钥,不是应用通用 APIKey",
    )
    if xfyun_apikey:
        st.session_state["xfyun_apikey"] = xfyun_apikey

    if st.session_state.get("xfyun_appid") and st.session_state.get("xfyun_apikey"):
        cfg = {
            "appid": st.session_state["xfyun_appid"],
            "apikey": st.session_state["xfyun_apikey"],
        }
        if cfg != _load_xfyun_settings():
            _save_xfyun_settings(cfg)
        st.caption("✅ 已记住你自己的讯飞 key(走云端实时转写)")
        if st.button("🗑️ 忘记已保存的讯飞 Key", key="forget_xfyun_key"):
            _XFYUN_SETTINGS_PATH.unlink(missing_ok=True)
            st.session_state["xfyun_appid"] = ""
            st.session_state["xfyun_apikey"] = ""
            st.rerun()

    st.divider()
    st.subheader("📚 历史记录")
    runs = list_runs()
    if not runs:
        st.caption("暂无历史记录,运行一次后自动存档。")
    else:
        for r in runs:
            with st.container(border=True):
                st.markdown(f"**{r['title'] or r['source_name']}**")
                st.caption(
                    f"{r['created_at']} · 讲话 {r['n_speakers'] or 0} 人 · "
                    f"{r['n_topics'] or 0} 议题 · {r['n_actions'] or 0} 待办"
                )
                b1, b2 = st.columns(2)
                b1.button(
                    "查看",
                    key=f"view_{r['id']}",
                    use_container_width=True,
                    on_click=_open_run,
                    args=(r["id"],),
                )
                if b2.button("删除", key=f"del_{r['id']}", use_container_width=True):
                    delete_run(r["id"])
                    st.rerun()


# Web Speech API 实时字幕组件(纯 HTML 前端,无构建步骤)。
# 浏览器内跑 SpeechRecognition,边录边把「已定稿」文本回传给 Python,
# 结束后再把最终全文回传;Python 侧据此同步跑 LLM 实时翻译。
_speech_recorder = components.declare_component(
    "speech_recorder",
    path=str(ROOT / "app" / "components" / "speech_recorder"),
)


@st.fragment(run_every="3s")
def _render_live_translation() -> None:
    """实时字幕「边听边译」面板:轮询最新转写,让 LLM 与转写同时工作。

    转写由浏览器 Web Speech API 负责(不受 Python 阻塞),这里每隔几秒
    把增长中的文本翻译一遍,实现字幕与译文同步出现。
    """
    target = TRANSLATE_TARGETS.get(translate_opt)
    if target is None:
        st.caption("在左侧开启「双语翻译」后,这里会随语音转写实时生成译文。")
        return
    if using_ollama and not _ollama_available():
        st.caption("本地 Ollama 未运行:请启动 Ollama,或在「⚙️ 设置」页「🔑 大模型」填入你自己的 key。")
        return
    src = (st.session_state.get("live_translate_src") or "").strip()
    if not src:
        st.caption("等待识别…说出内容后会实时翻译到这里。")
        return
    last = st.session_state.get("live_translate_last_src", "")
    if src != last:
        try:
            with st.spinner("🌐 实时翻译中…"):
                st.session_state["live_translate_result"] = translate(
                    src,
                    target=target,
                    glossary=glossary,
                    model=llm_model,
                    api_key=llm_api_key,
                    base_url=llm_base_url,
                )
        except Exception as exc:  # pragma: no cover - 网络/配额相关
            st.session_state["live_translate_result"] = f"(实时翻译失败:{exc})"
        st.session_state["live_translate_last_src"] = src
    result = st.session_state.get("live_translate_result", "")
    st.markdown("#### 🌐 实时译文")
    st.text(result or "翻译中…")
    if pop_fallback_notices():
        st.caption(f"📴 云端连不上,实时译文由本机 Ollama(`{settings.ollama_model}`)生成。")


def _render_dialogue(
    segments: list[dict],
    *,
    with_title: bool = True,
    with_timestamp: bool = True,
) -> None:
    """以对话气泡形式渲染转写稿,按讲话人着色。"""
    turns = merge_turns(segments)
    if not turns:
        st.info("没有可展示的对话内容。")
        return

    speakers: list[str] = []
    for t in turns:
        if t.get("speaker") and t["speaker"] not in speakers:
            speakers.append(t["speaker"])
    colors = {sp: SPEAKER_COLORS[i % len(SPEAKER_COLORS)] for i, sp in enumerate(speakers)}

    blocks = []
    for t in turns:
        sp = t.get("speaker") or "未识别"
        color = colors.get(sp, "#6b7280")
        text = _html.escape(t.get("text", ""))
        time_html = (
            f'<div style="flex:0 0 58px;font-size:12px;color:inherit;opacity:.5;padding-top:5px;">'
            f'{format_time(t.get("start", 0))}</div>'
            if with_timestamp
            else ""
        )
        blocks.append(
            '<div style="display:flex;gap:10px;margin:10px 0;align-items:flex-start;">'
            f"{time_html}"
            f'<div style="flex:0 0 78px;"><span style="background:{color};color:#fff;'
            f'padding:2px 9px;border-radius:10px;font-size:12px;white-space:nowrap;">'
            f'{_html.escape(sp)}</span></div>'
            f'<div style="flex:1;line-height:1.75;">{text}</div>'
            "</div>"
        )
    if with_title:
        st.caption(f"共 {len(turns)} 轮发言 · {len(speakers)} 位讲话人")
    with st.container(height=420, border=False):
        st.markdown("".join(blocks), unsafe_allow_html=True)


def _merge_turns_bilingual(segments: list[dict], *, max_gap: float = 3.0) -> list[dict]:
    """把同一说话人的连续分段合并成一"轮",原文与译文都一并拼接,保证双语对齐。"""
    turns: list[dict] = []
    for seg in segments:
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        translated = (seg.get("text_translated") or "").strip()
        speaker = seg.get("speaker")
        start, end = float(seg.get("start", 0.0)), float(seg.get("end", 0.0))
        if turns and turns[-1]["speaker"] == speaker and start - turns[-1]["end"] <= max_gap:
            turns[-1]["text"] += text
            if translated:
                turns[-1]["translated"] = (
                    f"{turns[-1]['translated']} {translated}".strip()
                )
            turns[-1]["end"] = end
        else:
            turns.append({
                "speaker": speaker,
                "start": start,
                "end": end,
                "text": text,
                "translated": translated,
            })
    return turns


def _render_dialogue_bilingual(
    segments: list[dict],
    *,
    with_title: bool = True,
    target_label: str = "译文",
    with_timestamp: bool = True,
) -> None:
    """双语对照渲染:每条发言上方原文、下方译文(仿会议同传 App 的样式)。"""
    turns = _merge_turns_bilingual(segments)
    if not turns:
        st.info("没有可展示的对话内容。")
        return

    speakers: list[str] = []
    for t in turns:
        sp = t.get("speaker")
        if sp and sp not in speakers:
            speakers.append(sp)
    colors = {sp: SPEAKER_COLORS[i % len(SPEAKER_COLORS)] for i, sp in enumerate(speakers)}
    has_speaker = bool(speakers)  # 纯文本输入无说话人时,不显示「未识别」标签

    blocks = []
    for t in turns:
        sp = t.get("speaker") or "未识别"
        color = colors.get(sp, "#6b7280")
        original = _html.escape(t.get("text", ""))
        translated = _html.escape(t.get("translated", ""))
        # 译文为空(翻译失败)时只显示原文,不留空行
        translated_html = (
            f'<div style="margin-top:6px;color:inherit;opacity:.65;line-height:1.7;">{translated}</div>'
            if translated
            else ""
        )
        speaker_html = (
            f'<div style="flex:0 0 78px;"><span style="background:{color};color:#fff;'
            f'padding:2px 9px;border-radius:10px;font-size:12px;white-space:nowrap;">'
            f'{_html.escape(sp)}</span></div>'
            if has_speaker
            else ""
        )
        blocks.append(
            '<div style="display:flex;gap:10px;margin:14px 0;align-items:flex-start;">'
            f'<div style="flex:0 0 58px;font-size:12px;color:inherit;opacity:.5;padding-top:5px;">'
            f'{format_time(t.get("start", 0)) if (has_speaker and with_timestamp) else ""}</div>'
            f"{speaker_html}"
            f'<div style="flex:1;line-height:1.75;">'
            f"<div>{original}</div>"
            f"{translated_html}"
            "</div>"
            "</div>"
        )
    if with_title:
        n_translated = sum(1 for t in turns if t.get("translated"))
        st.caption(
            f"共 {len(turns)} 轮发言 · {len(speakers)} 位讲话人 · "
            f"已译 {n_translated}/{len(turns)} 轮({target_label})"
        )
    with st.container(height=460, border=False):
        st.markdown("".join(blocks), unsafe_allow_html=True)


def _bilingual_to_markdown(segments: list[dict], *, with_timestamp: bool = True) -> str:
    """把逐段译文导出为「原文 / 译文」交替的 Markdown。"""
    turns = _merge_turns_bilingual(segments)
    lines = ["# 双语对照", ""]
    for t in turns:
        prefix = f"[{format_time(t['start'])}] " if (t.get("speaker") and with_timestamp) else ""
        speaker = f"{t['speaker']}:" if t.get("speaker") else ""
        lines.append(f"**{prefix}{speaker}** {t['text']}")
        if t.get("translated"):
            lines.append(f"> {t['translated']}")
        lines.append("")
    return "\n".join(lines)


def _markdown_paragraphs(text: str) -> str:
    """把多行文本整理成 markdown 段落:每个非空行之间用空行分隔。

    markdown 里单换行会被折叠成一行,只有空行才是真正的段落分隔;
    这里统一成空行,避免转写稿/译文整篇挤在一起。
    """
    blocks = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return "\n\n".join(blocks)


def _to_markdown(
    summary: dict,
    transcript_text: str,
    *,
    is_dialogue: bool = False,
    lang: str = "zh",
    brief: str = "",
) -> str:
    """把摘要渲染成 Markdown,便于答辩展示。lang="en" 时使用英文标题。"""
    L = {
        "zh": {"title": "会议纪要", "brief": "精简总结", "actions": "待办事项", "conclusion": "结论",
               "keywords": "关键词", "dialogue": "对话记录(按讲话人)", "transcript": "原始转录(校对后)",
               "owner": "负责人", "due": "截止", "join": "、"},
        "en": {"title": "Meeting Minutes", "brief": "Brief Summary", "actions": "Action Items", "conclusion": "Conclusion",
               "keywords": "Keywords", "dialogue": "Dialogue (by speaker)", "transcript": "Transcript (corrected)",
               "owner": "Owner", "due": "Due", "join": ", "},
    }[lang]
    lines = [f"# {summary.get('title') or L['title']}", ""]
    if brief:
        lines += [f"**{L['brief']}:** {brief}", ""]
    if summary.get("summary"):
        lines += [summary["summary"], ""]
    for topic in summary.get("topics", []) or []:
        lines.append(f"## {topic.get('title', '')}")
        for p in topic.get("points", []) or []:
            lines.append(f"- {p}")
        if topic.get("conclusion"):
            lines.append(f"\n**{L['conclusion']}:** {topic['conclusion']}\n")
    actions = summary.get("action_items", []) or []
    if actions:
        lines.append(f"## {L['actions']}")
        for a in actions:
            lines.append(
                f"- [ ] {a.get('task', '')}"
                f"{('(' + L['owner'] + ':' + a['owner'] + ')') if a.get('owner') else ''}"
                f"{('(' + L['due'] + ':' + a['deadline'] + ')') if a.get('deadline') else ''}"
            )
        lines.append("")
    if summary.get("keywords"):
        lines.append(f"**{L['keywords']}:** {L['join'].join(summary.get('keywords', []))}")
    heading = L["dialogue"] if is_dialogue else L["transcript"]
    # 转写/译文里的单换行在 markdown 里会被折叠,转成空行分隔才能看出段落
    lines += ["", "---", "", f"## {heading}", "", _markdown_paragraphs(transcript_text)]
    return "\n".join(lines)


def _qa_history(msgs: list[dict]) -> list[dict]:
    """把问答消息 [{role, content}] 折成 [{q, a}],只保留已完成的回合作为追问上下文。"""
    pairs = []
    i = 0
    while i < len(msgs):
        if (
            msgs[i].get("role") == "user"
            and i + 1 < len(msgs)
            and msgs[i + 1].get("role") == "assistant"
        ):
            pairs.append({"q": msgs[i]["content"], "a": msgs[i + 1]["content"]})
            i += 2
        else:
            i += 1
    return pairs


def _merge_terms(existing: str, terms: list[str]) -> str:
    """把新提取的术语并入已有术语表(逗号分隔),去重且保持原顺序。"""
    seen: set[str] = set()
    out: list[str] = []
    # 兼容用户手输的中文逗号
    for item in [*existing.replace("，", ",").split(","), *terms]:
        term = item.strip()
        if term and term.lower() not in seen:
            seen.add(term.lower())
            out.append(term)
    return ", ".join(out)


def _render_glossary_import(
    *, model: str, api_key: str, base_url: str, using_ollama: bool
) -> None:
    """「从文件导入术语」:抽文字 → LLM 提术语 → 合并进术语表输入框。

    不直接改 widget,而是把结果暂存到 session_state["_pending_glossary"],
    rerun 后由主流程在渲染输入框之前写入(Streamlit 只允许在 widget 创建前赋值)。
    """
    st.caption(
        "上传 PPT / Word / PDF / 文本,自动提取人名、机构、产品、术语,"
        "并入上方术语表,供后续纠错与翻译统一写法。"
    )
    up = st.file_uploader(
        "选择文件",
        type=["pptx", "docx", "pdf", "txt", "md", "csv"],
        key="glossary_file",
        help="旧版 .ppt / .doc 请先另存为 .pptx / .docx;扫描版 PDF 无文字层,读不出来",
    )
    if up is None:
        return

    data = up.getvalue()
    # 按文件内容哈希缓存解析结果,避免每次 rerun 重复解压
    sig = hashlib.md5(data).hexdigest()
    if st.session_state.get("glossary_file_sig") != sig:
        try:
            text = extract_text(up.name, data)
        except Exception as exc:
            st.error(f"读取文件失败:{exc}")
            return
        st.session_state["glossary_file_sig"] = sig
        st.session_state["glossary_file_text"] = text

    text = st.session_state.get("glossary_file_text", "")
    if not text:
        st.warning("文件里没读到文字(可能是纯图片幻灯片或扫描件)。")
        return

    st.caption(f"已读取 {len(text):,} 字。")
    if not st.button("✨ 提取术语并填入术语表", key="extract_terms_btn"):
        return

    if using_ollama and not _ollama_available():
        st.error(
            "本地 Ollama 未运行,无法提取术语;请在「⚙️ 设置」页填入你自己的 LLM key,"
            "或启动 Ollama 后重试。"
        )
        return
    try:
        with st.spinner("正在提取术语…"):
            terms = extract_terms(text, model=model, api_key=api_key, base_url=base_url)
    except Exception as exc:
        st.error(f"术语提取失败:{exc}")
        return
    if not terms:
        st.warning("没有提取到术语,可手动在术语表里填写。")
        return

    st.session_state["_pending_glossary"] = _merge_terms(
        st.session_state.get("glossary_input", ""), terms
    )
    st.rerun()


def _render_result(result: dict) -> None:
    """渲染一次运行(或历史记录)的完整结果:纪要 + 对话 + 导出。

    从 result dict 读取字段,故既可用于刚跑完的会话,也可用于回看历史。
    """
    summary = result.get("summary") or {}
    corrected = result.get("corrected_segments") or []
    source_segments = result.get("source_segments") or []
    corrected_text = result.get("corrected_text") or ""
    source_text = result.get("transcript") or ""
    source_name = result.get("source") or "未命名"
    n_speakers = len({s.get("speaker") for s in corrected if s.get("speaker")})
    translated_summary = result.get("translated_summary")
    translated_text = result.get("translated_text")
    translated_lang = result.get("translated_lang") or ""
    has_translation = bool(translated_summary or translated_text)
    md_lang = langs.markdown_lang(translated_lang)
    show_time = bool(result.get("has_timestamps", True))
    brief = result.get("brief") or ""

    st.divider()
    if result.get("fallback_local"):
        st.warning(
            "⚠️ 云端大模型连不上(断网或服务不可达),本次已自动改用本机 Ollama"
            f"(`{settings.ollama_model}`)。纪要质量会低于云端模型;"
            "联网后重新生成即可恢复正常。"
        )
    if brief:
        st.subheader("📌 精简总结")
        st.markdown(f'<div class="mc-brief">{_html.escape(brief)}</div>', unsafe_allow_html=True)
    st.subheader("📋 结构化纪要")

    c1, c2 = st.columns([3, 1])
    with c1:
        if summary.get("title"):
            st.markdown(f"### {summary['title']}")
        if summary.get("summary"):
            st.markdown(summary["summary"])
    with c2:
        if n_speakers:
            st.metric("讲话人数", n_speakers)
        st.metric("转写段数", len(source_segments))
        st.metric("议题数", len(summary.get("topics", []) or []))
        st.metric("待办数", len(summary.get("action_items", []) or []))

    # 讲话人概览
    if n_speakers:
        st.markdown("#### 👥 讲话人概览")
        stats = speaker_stats(corrected)
        cols = st.columns(min(len(stats), 4))
        for i, item in enumerate(stats):
            if show_time:
                mins = int(item["duration"] // 60)
                secs = int(item["duration"] % 60)
                value, delta = f"{mins}分{secs:02d}秒", f"{item['segments']} 句"
            else:  # 文本/字幕无真实时长,只统计发言句数
                value, delta = f"{item['segments']} 句", ""
            cols[i % len(cols)].metric(item["speaker"], value, delta)

    # 议题
    topics = summary.get("topics", []) or []
    if topics:
        st.markdown("#### 🗂️ 议题与结论")
        for t in topics:
            title = t.get("title") or "未命名议题"
            with st.expander(f"**{title}**", expanded=False):
                for p in t.get("points", []) or []:
                    st.markdown(f"- {p}")
                if t.get("conclusion"):
                    st.markdown(f"> **结论:** {t['conclusion']}")

    # 待办
    actions = summary.get("action_items", []) or []
    if actions:
        st.markdown("#### ✅ 待办事项")
        rows = []
        for a in actions:
            rows.append({
                "任务": a.get("task", ""),
                "负责人": a.get("owner") or "—",
                "截止": a.get("deadline") or "—",
            })
        st.table(rows)

    # 关键词
    keywords = summary.get("keywords", []) or []
    if keywords:
        st.markdown("#### 🏷️ 关键词")
        st.markdown(" ".join(f"`{k}`" for k in keywords))

    # 双语翻译(可选)
    if has_translation:
        st.markdown(f"#### 🌐 双语纪要({translated_lang})")
        trans_summary = translated_summary or summary
        trans_text = translated_text or (
            render_dialogue(corrected, with_timestamp=show_time) if n_speakers else corrected_text
        )
        with st.expander(f"查看{translated_lang}版纪要", expanded=False):
            st.markdown(
                _to_markdown(trans_summary, trans_text, is_dialogue=bool(n_speakers), lang=md_lang)
            )

    # 校对稿 / 原始转写 / 翻译 / 导出
    st.divider()
    detail_tab, raw_tab, trans_tab, export_tab = st.tabs(
        ["📄 校对稿", "📄 原始转写", "🌐 翻译", "⬇️ 导出"]
    )

    with detail_tab:
        has_seg_translation = any(s.get("text_translated") for s in corrected)
        if has_seg_translation:
            seg_sig = hashlib.md5(
                (source_name + corrected_text[:200]).encode("utf-8")
            ).hexdigest()[:8]
            view = st.radio(
                "显示方式",
                ["🌐 双语对照", "📄 仅原文"],
                horizontal=True,
                key=f"biview_{seg_sig}",
            )
            if view.startswith("📄"):
                if n_speakers:
                    _render_dialogue(corrected, with_timestamp=show_time)
                else:
                    st.text_area("校对后全文", corrected_text, height=280)
            else:
                _render_dialogue_bilingual(
                    corrected, target_label=translated_lang or "译文", with_timestamp=show_time
                )
        elif n_speakers:
            _render_dialogue(corrected, with_timestamp=show_time)
        else:
            st.text_area("校对后全文", corrected_text, height=280)

    with raw_tab:
        if n_speakers:
            _render_dialogue(source_segments, with_title=False, with_timestamp=show_time)
        else:
            st.text_area("原始转写全文", source_text, height=280)

    with trans_tab:
        # 结果页内的「即时翻译」:随时把校对稿或纪要翻成另一种语言,
        # 不必回侧边栏重跑整条流程。译文缓存在 session_state,重复点击/下载不再重复请求。
        dialog_src = (
            render_dialogue(corrected, with_timestamp=False) if n_speakers else corrected_text
        )
        sig = hashlib.md5(
            (source_name + (corrected_text or source_text or "")).encode("utf-8")
        ).hexdigest()[:8]
        c1, c2 = st.columns([2, 1])
        scope = c1.radio(
            "翻译内容",
            ["校对稿全文", "结构化纪要"],
            horizontal=True,
            key=f"tscope_{sig}",
        )
        lang_choice = c2.selectbox(
            "目标语言",
            langs.labels(),
            key=f"tlang_{sig}",
            help="源语言自动识别,可中译外,也可外译中",
        )
        target = langs.code_of(lang_choice) or langs.DEFAULT
        cache_key = f"trans_{sig}_{scope}_{target}"

        if st.button("🌐 开始翻译", key=f"tbtn_{sig}"):
            if scope == "结构化纪要":
                if not summary:
                    st.warning("没有可翻译的纪要。")
                else:
                    try:
                        with st.spinner(f"正在翻译纪要成{lang_choice}…"):
                            st.session_state[cache_key] = {
                                "type": "summary",
                                "value": translate_summary(
                                    summary,
                                    target=target,
                                    glossary=glossary,
                                    model=llm_model,
                                    api_key=llm_api_key,
                                    base_url=llm_base_url,
                                ),
                            }
                    except Exception as exc:  # pragma: no cover - 网络/配额相关
                        st.error(f"翻译失败:{exc}")
            else:
                if not dialog_src.strip():
                    st.warning("没有可翻译的文本。")
                else:
                    try:
                        with st.spinner(f"正在翻译成{lang_choice}…"):
                            st.session_state[cache_key] = {
                                "type": "text",
                                "value": translate(
                                    dialog_src,
                                    target=target,
                                    glossary=glossary,
                                    model=llm_model,
                                    api_key=llm_api_key,
                                    base_url=llm_base_url,
                                ),
                            }
                    except Exception as exc:  # pragma: no cover - 网络/配额相关
                        st.error(f"翻译失败:{exc}")

        cached = st.session_state.get(cache_key)
        if cached and cached.get("type") == "summary":
            md = _to_markdown(
                cached["value"],
                dialog_src,
                is_dialogue=bool(n_speakers),
                lang=langs.markdown_lang(target),
            )
            st.markdown(f"#### {lang_choice}版纪要")
            st.markdown(md)
            st.download_button(
                f"下载{lang_choice}纪要 Markdown",
                data=md,
                file_name=f"{source_name}.{target}.md",
                mime="text/markdown",
                key=f"tdl_{cache_key}",
            )
        elif cached:
            st.markdown(f"#### {lang_choice}译文")
            st.text_area(
                "译文(可复制)",
                cached["value"],
                height=300,
                key=f"tout_{cache_key}",
            )
            st.download_button(
                f"下载{lang_choice}译文",
                data=cached["value"],
                file_name=f"{source_name}.{target}.txt",
                mime="text/plain",
                key=f"tdlt_{cache_key}",
            )
        else:
            st.caption(
                "选择内容与目标语言后点「开始翻译」。源语言自动识别,中译外 / 外译中都可以;"
                "翻译优先走百度 API(省 token),未配置或该语种不支持时回退 LLM。"
            )

    with export_tab:
        export_data = {
            "source": source_name,
            "summary": summary,
            "brief": brief,
            "transcript": source_text,
            "corrected_text": corrected_text,
            "corrected_segments": corrected,
            "dialogue": render_dialogue(corrected, with_timestamp=show_time) if n_speakers else None,
            "speaker_stats": speaker_stats(corrected) if n_speakers else None,
            "translated_summary": translated_summary,
            "translated_text": translated_text,
            "translated_lang": translated_lang,
        }
        st.download_button(
            "下载 JSON 结果",
            data=json.dumps(export_data, ensure_ascii=False, indent=2),
            file_name=f"{source_name}.summary.json",
            mime="application/json",
        )
        st.download_button(
            "下载纪要 Markdown",
            data=_to_markdown(
                summary,
                render_dialogue(corrected, with_timestamp=show_time) if n_speakers else corrected_text,
                is_dialogue=bool(n_speakers),
                brief=brief,
            ),
            file_name=f"{source_name}.md",
            mime="text/markdown",
        )
        if any(s.get("text_translated") for s in corrected):
            st.download_button(
                "下载双语对照 Markdown",
                data=_bilingual_to_markdown(corrected, with_timestamp=show_time),
                file_name=f"{source_name}.bilingual.md",
                mime="text/markdown",
            )
        if has_translation:
            trans_summary = translated_summary or summary
            trans_text = translated_text or (
                render_dialogue(corrected, with_timestamp=show_time) if n_speakers else corrected_text
            )
            lang_suffix = langs.iso_of(translated_lang)
            st.download_button(
                f"下载{translated_lang}纪要 Markdown",
                data=_to_markdown(trans_summary, trans_text, is_dialogue=bool(n_speakers), lang=md_lang),
                file_name=f"{source_name}.{lang_suffix}.md",
                mime="text/markdown",
            )

    # ===== 会议问答:就会议内容提问,或按需生成总结 / 纪要 =====
    st.divider()
    st.subheader("💬 会议问答")
    st.caption("就本场会议提问,或让我按你的要求生成总结、纪要、待办清单。")

    qa_sig = hashlib.md5(
        (source_name + corrected_text).encode("utf-8")
    ).hexdigest()[:8]
    qa_key = f"qa_{qa_sig}"
    if qa_key not in st.session_state:
        st.session_state[qa_key] = []

    for msg in st.session_state[qa_key]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if qa_prompt := st.chat_input("就会议内容提问…", key=f"qa_input_{qa_sig}"):
        st.session_state[qa_key].append({"role": "user", "content": qa_prompt})
        with st.chat_message("user"):
            st.markdown(qa_prompt)
        context = (
            render_dialogue(corrected, with_timestamp=False)
            if n_speakers
            else corrected_text
        )
        with st.chat_message("assistant"):
            with st.spinner("思考中…"):
                try:
                    answer = ask_meeting(
                        context,
                        qa_prompt,
                        history=_qa_history(st.session_state[qa_key]),
                        model=llm_model,
                        api_key=llm_api_key,
                        base_url=llm_base_url,
                    )
                except Exception as exc:  # pragma: no cover - 网络/配额相关
                    answer = f"⚠️ 问答失败:{exc}"
            st.markdown(answer)
            if pop_fallback_notices():
                st.caption(
                    f"📴 云端大模型连不上,此回答由本机 Ollama(`{settings.ollama_model}`)生成。"
                )
        st.session_state[qa_key].append({"role": "assistant", "content": answer})


st.set_page_config(page_title="会议智治", page_icon="🎙️", layout="wide")
_inject_css()

# ============ 从磁盘恢复用户配置(主界面 / 设置页共用)============
if "llm_cfg_seeded" not in st.session_state:
    st.session_state["llm_cfg_seeded"] = True
    saved = _load_llm_settings()
    st.session_state.setdefault("llm_api_key", saved.get("api_key", ""))
    provider = saved.get("provider", "DeepSeek · deepseek-chat")
    if provider not in _LLM_PRESETS:
        provider = "DeepSeek · deepseek-chat"
    st.session_state.setdefault("llm_provider", provider)
    st.session_state.setdefault("llm_custom_url", saved.get("base_url") or "https://api.deepseek.com")
    st.session_state.setdefault("llm_custom_model", saved.get("model") or "deepseek-chat")
if "xfyun_cfg_seeded" not in st.session_state:
    st.session_state["xfyun_cfg_seeded"] = True
    saved = _load_xfyun_settings()
    st.session_state.setdefault("xfyun_appid", saved.get("appid", ""))
    st.session_state.setdefault("xfyun_apikey", saved.get("apikey", ""))

# 无论停在哪个导航页,都先把本次运行要用的 LLM / 讯飞配置算好
llm_api_key, llm_base_url, llm_model, using_ollama = _llm_runtime_config()
xfyun_appid = st.session_state.get("xfyun_appid", "")
xfyun_apikey = st.session_state.get("xfyun_apikey", "")

# ============ 侧边栏配置 ============
with st.sidebar:
    st.session_state.setdefault("nav", "📝 主界面")
    nav = st.radio(
        "页面",
        ["📝 主界面", "⚙️ 设置"],
        horizontal=True,
        label_visibility="collapsed",
        key="nav",
    )
    st.caption(
        "🔑 大模型 / 讯飞转写 key 请在「⚙️ 设置」页填你自己的;"
        "留空则回退本地 Ollama / 浏览器语音。"
    )
    st.header("⚙️ 配置")
    # 文件导入的术语在这里落地:必须在 text_input 创建之前写入 session_state,
    # Streamlit 只允许在 widget 创建前赋值(创建后赋值会抛 StreamlitAPIException)。
    _pending_glossary = st.session_state.pop("_pending_glossary", None)
    if _pending_glossary is not None:
        st.session_state["glossary_input"] = _pending_glossary
    glossary = st.text_input(
        "术语表",
        key="glossary_input",
        placeholder="余华, 苏童, 莫言, DeepSeek",
        help="逗号分隔的专有名词,用于纠正音译错误、统一写法;也可从文件一键导入",
    )
    with st.expander("📎 从文件导入术语", expanded=False):
        _render_glossary_import(
            model=llm_model,
            api_key=llm_api_key,
            base_url=llm_base_url,
            using_ollama=using_ollama,
        )
    lang_opt = st.selectbox("语言", ["自动检测", "中文 (zh)", "英文 (en)"])
    translate_opt = st.selectbox(
        "双语翻译",
        ["不翻译", *TRANSLATE_TARGETS],
        index=0,
        help="开启后生成逐句双语对照(原文 + 译文),并把纪要和转写稿翻译成目标语言;"
        "支持中、英、日、韩、法、德、西、葡、意、俄、阿、泰、越等语种互译",
    )
    enable_correct = st.checkbox("LLM 同音字纠错", value=True, help="对转写结果做同音字/术语修正")
    enable_diarize = st.checkbox(
        "说话人分离",
        value=True,
        help="识别不同讲话人,按「发言人」分段呈现;仅对音频有效",
    )
    num_speakers = st.number_input(
        "说话人数(0 = 自动判断)",
        min_value=0,
        max_value=20,
        value=0,
        step=1,
        disabled=not enable_diarize,
    )
    enable_speaker_llm = st.checkbox(
        "LLM 说话人分析(文本/字幕)",
        value=True,
        help="粘贴文本 / 实时字幕没有声纹信息,改用 LLM 判断对话里的说话人并分段;仅对文本与实时字幕输入有效",
    )
    asr_model = st.selectbox("ASR 模型", ["tiny", "base", "small", "medium"], index=2)
    st.caption("首次运行会自动下载模型(small 约 460MB)。")

language = {"自动检测": None, "中文 (zh)": "zh", "英文 (en)": "en"}[lang_opt]

# ============ 标题(主界面 / 设置页共用) ============
st.title("🎙️ 会议智治与跨语言沟通平台")

# ============ 输入区标签页 ============
# 顺序即界面顺序;常驻录音卡片只挂在「🎙️ 实时字幕」页(见下方 CSS 的 data-key)。
_INPUT_TABS = ["🎧 上传音频", "🎙️ 实时字幕", "🎤 本地录音", "📝 粘贴文本"]
_LIVE_TAB_INDEX = _INPUT_TABS.index("🎙️ 实时字幕")

# ============ 常驻实时字幕组件 ============
# 关键:无论停在哪个导航页都保持挂载。切换「⚙️ 设置」页填 key 时,若组件只渲染在
# 「实时字幕」标签里,Streamlit 会因 st.stop() 跳过该标签而卸载 iframe,正在进行的
# 录音 / 转写随之被打断。因此把它提到页面顶部、每个 rerun 都渲染一次,iframe 常驻
# 不销毁;设置页 / 历史回看仅用 CSS 隐藏该卡片,录音仍在后台继续。
lang_code = {"自动检测": "zh-CN", "中文 (zh)": "zh-CN", "英文 (en)": "en-US"}[lang_opt]
cloud_enabled = bool(xfyun_appid and xfyun_apikey)
raw = _speech_recorder(
    lang=lang_code,
    key="speech_recorder",
    cloud_enabled=cloud_enabled,
    backend_url=settings.rtasr_ws_url,
    xfyun_appid=xfyun_appid,
    xfyun_apikey=xfyun_apikey,
)

# 录音卡片只在「🎙️ 实时字幕」标签页可见:其它标签页 / 设置页 / 历史回看一律 CSS 隐藏,
# 但 iframe 仍留在 DOM 里(display:none 不会卸载组件)。
# 例外:录音进行中一律显示 —— 被隐藏的跨域 iframe 会被浏览器节流,组件里 40ms 的音频帧
# 定时器一旦被拖慢,讯飞实时转写就会卡住(Web Speech 路径不受影响)。
# 判断方式:Streamlit 1.66 的标签页按钮带 aria-selected 与 data-key(等于标签下标);
# 页面没有标签页时(设置页 / 历史回看):has() 不成立,同样隐藏。需 Chrome / Edge 105+,
# 与 Web Speech API 的浏览器要求一致。
recording = bool(isinstance(raw, dict) and raw.get("running"))
if not recording:
    st.markdown(
        "<style>"
        f'body:not(:has([role="tab"][aria-selected="true"][data-key="{_LIVE_TAB_INDEX}"])) '
        '[data-testid="stCustomComponentV1"]{display:none;}'
        "</style>",
        unsafe_allow_html=True,
    )

# ============ 设置页 ============
if nav == "⚙️ 设置":
    _render_settings()
    st.stop()

st.caption("上传会议录音或粘贴对话文本,一键生成结构化纪要")

# ============ 历史记录查看 ============
if st.session_state.get("view_run_id"):
    data = get_run(st.session_state["view_run_id"])
    if data:
        st.subheader(f"📚 历史记录 · {data.get('source', '')}")
        _render_result(data)
        if st.button("← 返回新建纪要", type="secondary"):
            st.session_state.pop("view_run_id", None)
            st.rerun()
    else:
        st.warning("该记录已被删除。")
        st.session_state.pop("view_run_id", None)
        st.rerun()
    st.stop()

# ============ 输入区 ============
tab_audio, tab_live, tab_mic, tab_text = st.tabs(_INPUT_TABS)

source_segments: list[dict] | None = None
source_text: str = ""
source_name: str = ""
# 上传 / 录音统一抽象为「原始音频字节 + 后缀」,下游走同一条转写链路。
audio_data: bytes | None = None
audio_suffix: str = ".wav"

with tab_audio:
    uploaded = st.file_uploader(
        "选择音频文件",
        type=["wav", "mp3", "m4a", "flac", "ogg", "opus", "mp4", "webm", "aac"],
        help="支持常见音频/视频格式,内部自动转 16k",
    )
    if uploaded is not None:
        source_name = Path(uploaded.name).stem
        audio_data = uploaded.getvalue()
        audio_suffix = Path(uploaded.name).suffix
        st.audio(uploaded)

with tab_live:
    # 实时字幕:录音 / 转写由页面顶部的常驻组件负责,这里只承接它回传的结果。
    st.caption(
        "实时语音转写:侧边栏填了自己的讯飞 key 走云端(接近讯飞听见);"
        "留空则回退浏览器 Web Speech API(需 Chrome / Edge 并联网)。"
    )
    st.session_state.setdefault("wsr_edit", "")
    st.session_state.setdefault("wsr_last", "")

    # 组件回传 {text, running}:running=true 表示录音仍在进行,text 为已定稿文本。
    if isinstance(raw, dict):
        live_raw_text = (raw.get("text") or "").strip()
    else:  # 兼容旧组件返回纯字符串
        live_raw_text = (raw or "").strip()

    # 边录边把「已定稿」文本喂给实时翻译(LLM 与转写同时工作)
    if live_raw_text:
        st.session_state["live_translate_src"] = live_raw_text

    # 仅在「新识别结果」到达时覆盖文本框,避免吞掉用户后续的手动修改
    if live_raw_text and live_raw_text != st.session_state["wsr_last"]:
        st.session_state["wsr_last"] = live_raw_text
        st.session_state["wsr_edit"] = live_raw_text

    col_text, col_trans = st.columns([1, 1])
    with col_text:
        live_text = st.text_area(
            "实时转写结果(可编辑)",
            key="wsr_edit",
            height=260,
            placeholder="点击上方「开始录音」,识别文字会实时显示并回填到这里…",
        )
    with col_trans:
        _render_live_translation()

    if live_text.strip():
        source_text = live_text.strip()
        source_name = "实时字幕"

with tab_mic:
    mic_audio = st.audio_input(
        "🎤 点击开始录音",
        help="使用浏览器麦克风录音(16kHz),结束后点击下方按钮用本地 faster-whisper 转写",
    )
    if mic_audio is not None:
        source_name = "本地录音"
        audio_data = mic_audio.getvalue()
        audio_suffix = ".wav"
        st.audio(mic_audio)
        st.success("录音已就绪,点击下方「开始生成纪要」即可本地转写。")
    else:
        st.caption("全本地转写,不上传外部服务;无需联网,适合隐私敏感场景。")

with tab_text:
    text_input = st.text_area(
        "粘贴会议 / 对话文本",
        height=220,
        placeholder="在这里粘贴转写稿或对话内容,直接跳过语音识别...",
    )
    if text_input.strip():
        source_text = text_input.strip()
        source_name = "文本输入"

run = st.button("🚀 开始生成纪要", type="primary", use_container_width=True)

# ============ 处理流程 ============
if run:
    if audio_data is None and not source_text:
        st.warning("请先上传音频、录音、实时字幕或粘贴文本。")
        st.stop()

    if using_ollama and not _ollama_available():
        st.error(
            "本地 Ollama 未运行或未拉取模型:请安装并启动 Ollama(ollama.com)后执行 "
            f"`ollama pull {settings.ollama_model}`,或在「⚙️ 设置」页「🔑 大模型」填入你自己的 key。"
        )
        st.stop()

    # 清掉上一轮的回退记录,本轮是否有回退由下面重新收集
    clear_fallback_notices()
    duration = None
    # 四步状态机:idle / running / done / failed,实时反映到流水线卡片
    states: dict[str, tuple[str, str]] = {k: ("idle", "") for k, _, _ in _PIPELINE_STEPS}
    if audio_data is None:
        states["asr"] = ("done", "文本输入,跳过")
        states["diar"] = (
            ("running", "LLM 说话人分析…") if enable_speaker_llm else ("done", "已关闭")
        )

    pipeline_ph = st.empty()
    live_ph = st.empty()

    def draw_pipeline() -> None:
        pipeline_ph.markdown(_pipeline_html(states), unsafe_allow_html=True)

    draw_pipeline()

    # ===== 阶段 1:语音转写 ∥ 说话人识别(两个本地模型同时跑)=====
    spans: list[dict] | None = None
    live_segments: list[dict] = []
    tmp_path: str | None = None
    try:
        if audio_data is not None:
            with tempfile.NamedTemporaryFile(suffix=audio_suffix, delete=False) as tmp:
                tmp.write(audio_data)
                tmp_path = tmp.name

            # 先探一次时长:说话人分离要用它判断时间戳单位,而它不等 ASR 结束
            duration_est = probe_duration(tmp_path)
            states["asr"] = ("running", "加载模型…")
            if enable_diarize:
                states["diar"] = ("running", "与转写同时进行…")
            draw_pipeline()

            def _render_live(seg=None) -> None:
                recent = live_segments[-8:]
                more = len(live_segments) - len(recent)
                if seg is None:
                    head = (
                        '<div class="mc-live-head"><span style="color:#10b981;font-weight:600;">'
                        f"✅ 转写完成</span><span style=\"color:inherit;opacity:.65;\">共 {len(live_segments)} 段</span></div>"
                    )
                else:
                    total = (
                        f" / {format_time(duration_est)} "
                        f"({min(seg.end / duration_est, 1) * 100:.0f}%)"
                        if duration_est
                        else ""
                    )
                    head = (
                        '<div class="mc-live-head"><span style="color:#3b82f6;font-weight:600;">'
                        '<span class="mc-live-dot"></span>'
                        f"实时转写中</span><span style=\"color:inherit;opacity:.65;\">"
                        f"{len(live_segments)} 段 · {format_time(seg.end)}{total}</span>"
                        + (
                            f'<span style="color:inherit;opacity:.5;">(上方省略 {more} 段)</span>'
                            if more > 0
                            else ""
                        )
                        + "</div>"
                    )
                rows = []
                for i, s in enumerate(recent):
                    is_latest = seg is not None and i == len(recent) - 1
                    cls = "mc-live-row latest" if is_latest else "mc-live-row"
                    cursor = '<span class="mc-live-cursor"></span>' if is_latest else ""
                    rows.append(
                        f'<div class="{cls}"><span class="mc-live-time">'
                        f'{format_time(s["start"])}</span>'
                        f'<span class="mc-live-text">{_html.escape(s["text"])}{cursor}</span></div>'
                    )
                rows = "".join(rows)
                live_ph.markdown(
                    f'<div class="mc-live">{head}{rows}</div>', unsafe_allow_html=True
                )

            def on_segment(seg):
                live_segments.append({"start": seg.start, "end": seg.end, "text": seg.text})
                _render_live(seg)

            # 说话人识别丢到后台线程,与主线程的转写并行
            pool = ThreadPoolExecutor(max_workers=1)
            try:
                diar_future = (
                    pool.submit(
                        diarize,
                        tmp_path,
                        num_speakers=int(num_speakers) or None,
                        duration=duration_est,
                    )
                    if enable_diarize
                    else None
                )

                prompt = build_initial_prompt(language, glossary)
                transcript = transcribe_file(
                    tmp_path,
                    language=language,
                    initial_prompt=prompt,
                    model_size=asr_model,
                    on_segment=on_segment,
                )
                duration = transcript.duration
                # 繁转简,避免 Whisper 混入繁体字
                source_segments = segments_to_simplified(
                    [s for s in transcript.to_dict()["segments"]]
                )
                source_text = to_simplified(transcript.text)
                if live_segments:
                    _render_live()
                states["asr"] = (
                    "done",
                    f"{len(source_segments)} 段 · {format_time(duration)} · "
                    f"{transcript.elapsed_seconds:.0f}s",
                )
                draw_pipeline()

                # 收回说话人识别结果(通常此时已跑完,无需等待)
                if diar_future is not None:
                    try:
                        spans = diar_future.result()
                    except Exception as exc:  # pragma: no cover - 模型/依赖相关
                        spans = None
                        states["diar"] = ("failed", f"识别失败,已跳过:{str(exc)[:40]}")
                    else:
                        source_segments = assign_speakers(source_segments, spans)
                        n_people = len(
                            {s["speaker"] for s in source_segments if s.get("speaker")}
                        )
                        states["diar"] = ("done", f"{n_people} 位讲话人")
                    draw_pipeline()
                else:
                    states["diar"] = ("done", "已关闭")
                    draw_pipeline()
            finally:
                pool.shutdown(wait=False)
        elif source_text:
            source_text = to_simplified(source_text)
            if enable_speaker_llm:
                draw_pipeline()
                try:
                    source_segments = annotate_speakers(
                        source_text,
                        model=llm_model,
                        api_key=llm_api_key,
                        base_url=llm_base_url,
                    )
                    n_people = len(
                        {s.get("speaker") for s in source_segments if s.get("speaker")}
                    )
                    states["diar"] = ("done", f"{n_people} 位讲话人")
                except Exception as exc:  # pragma: no cover - 网络/结构异常
                    source_segments = [{"start": 0.0, "end": 0.0, "text": source_text}]
                    states["diar"] = ("failed", f"分析失败,按单人处理:{str(exc)[:30]}")
                draw_pipeline()
            else:
                source_segments = [{"start": 0.0, "end": 0.0, "text": source_text}]

        # ===== 阶段 2:同音字纠错(后续步骤都依赖它)=====
        if enable_correct:
            states["correct"] = ("running", "同音字 + 术语统一…")
            draw_pipeline()
            corrected = segments_to_simplified(
                correct_transcript(
                    source_segments,
                    glossary=glossary,
                    model=llm_model,
                    api_key=llm_api_key,
                    base_url=llm_base_url,
                )
            )
            corrected_text = to_simplified("".join(s.get("text", "") for s in corrected))
            states["correct"] = ("done", f"{len(corrected)} 段已校对")
        else:
            corrected = source_segments
            corrected_text = source_text
            states["correct"] = ("done", "已关闭")
        draw_pipeline()

        # ===== 阶段 3:结构化纪要 ∥ 逐段翻译(两次 LLM 调用同时发出)=====
        has_speaker = any(s.get("speaker") for s in corrected)
        summary_input = (
            render_dialogue(corrected, with_timestamp=False) if has_speaker else corrected_text
        )

        target: str | None = None
        translated_lang = ""
        if translate_opt != "不翻译":
            target = TRANSLATE_TARGETS.get(translate_opt)
            translated_lang = langs.label_of(target)

        states["llm"] = (
            "running",
            "纪要 + 翻译并行中…" if target else "生成结构化纪要…",
        )
        draw_pipeline()

        pool2 = ThreadPoolExecutor(max_workers=3)
        try:
            sum_future = pool2.submit(
                summarize,
                summary_input,
                model=llm_model,
                api_key=llm_api_key,
                base_url=llm_base_url,
            )
            brief_future = pool2.submit(
                brief_summary,
                summary_input,
                model=llm_model,
                api_key=llm_api_key,
                base_url=llm_base_url,
            )
            if target:
                # 有说话人 → 逐段翻译(供双语对照);无说话人 → 全文直译,
                # 交给 LLM 按上下文与长度分自然段落,避免逐句拼接挤成一段
                if has_speaker:
                    tr_future = pool2.submit(
                        translate_segments,
                        corrected,
                        target=target,
                        glossary=glossary,
                        model=llm_model,
                        api_key=llm_api_key,
                        base_url=llm_base_url,
                    )
                else:
                    tr_future = pool2.submit(
                        translate,
                        corrected_text,
                        target=target,
                        glossary=glossary,
                        model=llm_model,
                        api_key=llm_api_key,
                        base_url=llm_base_url,
                    )
            else:
                tr_future = None
            summary = deep_to_simplified(sum_future.result())
            try:
                brief = to_simplified(brief_future.result() or "")
            except Exception:  # pragma: no cover - 网络/配额相关,精简总结失败不阻断主流程
                brief = ""
            translation = tr_future.result() if tr_future else None
        finally:
            pool2.shutdown(wait=False)

        translated_summary = None
        translated_text = None
        if translation is not None:
            states["llm"] = ("running", f"翻译纪要成{translated_lang}…")
            draw_pipeline()
            translated_summary = translate_summary(
                summary,
                target=target,
                glossary=glossary,
                model=llm_model,
                api_key=llm_api_key,
                base_url=llm_base_url,
            )
            if has_speaker:
                # 逐段译文:既挂到每段上做双语对照,又拼成整篇对话译文
                corrected = [
                    {
                        **s,
                        "text_translated": (
                            translation[i] if i < len(translation) else ""
                        ),
                    }
                    for i, s in enumerate(corrected)
                ]
                translated_text = render_dialogue(
                    [
                        {
                            "start": s.get("start", 0.0),
                            "end": s.get("end", 0.0),
                            "speaker": s.get("speaker"),
                            "text": s.get("text_translated", ""),
                        }
                        for s in corrected
                    ],
                    with_timestamp=False,
                )
            else:
                # 全文直译的结果已按上下文与长度分段,直接作为整篇译文
                translated_text = translation
        states["llm"] = (
            "done",
            f"{len(summary.get('topics', []) or [])} 议题 · "
            f"{len(summary.get('action_items', []) or [])} 待办"
            + (f" · 已译{translated_lang}" if translated_lang else ""),
        )
        draw_pipeline()
    finally:
        # 临时音频用完即删(无论成功失败都清理)
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)

    # 存档 + 展示(结果 dict 同时供历史记录复用)
    # 本轮若发生过「云端不可达 → 回退本机 Ollama」,记入结果以便结果页显式提示
    fallback_local = bool(pop_fallback_notices())
    result = {
        "source": source_name,
        "duration": duration,
        "summary": summary,
        "brief": brief,
        "transcript": source_text,
        "source_segments": source_segments,
        "corrected_text": corrected_text,
        "corrected_segments": corrected,
        "translated_summary": translated_summary,
        "translated_text": translated_text,
        "translated_lang": translated_lang,
        "has_timestamps": audio_data is not None,
        "fallback_local": fallback_local,
    }
    save_run(source_name, result)
    st.session_state["last_result"] = result
    _render_result(result)

elif st.session_state.get("last_result"):
    # 结果页内的交互(翻译 / 问答 / 下载)会触发 rerun,而「开始生成纪要」按钮已复位;
    # 这里从 session_state 恢复上一次结果,保证结果页在 rerun 后不消失。
    _render_result(st.session_state["last_result"])
