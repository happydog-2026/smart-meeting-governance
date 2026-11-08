"""新版网页界面(FastAPI)接口层。

index.html 由本模块的 `/` 直接返回;页面里的上传 / 转写 / 纪要 / 翻译 / 问答 /
导出 / 设置全部走这里的 /api/* 接口,复用 app/services 下同一套模型与存档。

设计要点:
- 处理流水线耗时较长(ASR + 说话人分离 + 多次 LLM 调用),用后台线程 + 任务轮询,
  不阻塞 FastAPI 事件循环,也不占用 uvicorn 工作进程。
- 设置文件与 Streamlit 版共用(data/llm_settings.json、data/xfyun_settings.json),
  两个界面填的 key 互相生效。
- 历史记录直接复用 app/services/history.py,与 Streamlit 版共享 data/history.db。
"""

from __future__ import annotations

import json
import re
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, Response
from pydantic import BaseModel

from app.config import ROOT, settings
from app.services import langs, render as render_svc
from app.services.asr import build_initial_prompt, transcribe_file
from app.services.auth import (
    SESSION_COOKIE,
    SESSION_DAYS,
    authenticate,
    backfill_orphan_runs,
    count_users,
    create_guest,
    create_session,
    create_user,
    delete_session,
    get_user_by_token,
    purge_user,
    upgrade_guest,
)
from app.services.audio import probe_duration
from app.services.diarize import diarize
from app.services.extract import extract_text
from app.services.history import (
    TRASH_RETENTION_DAYS,
    cancel_share,
    create_share,
    delete_run,
    get_run,
    get_shared_run,
    list_deleted_runs,
    list_favorite_runs,
    list_runs,
    list_shared_runs,
    purge_run,
    restore_run,
    save_run,
    set_favorite,
)
from app.services.share_page import render_share_page
from app.services.llm import (
    annotate_speakers,
    ask_meeting,
    brief_summary,
    clear_fallback_notices,
    correct_transcript,
    extract_terms,
    pop_fallback_notices,
    summarize,
)
from app.services.speaker import assign_speakers, render_dialogue
from app.services.text import deep_to_simplified, segments_to_simplified, to_simplified
from app.services.translate import translate, translate_segments, translate_summary

router = APIRouter()

# ============ 本地配置(每账号一份;旧版全局文件仅用于一次性迁移) ============
_DATA = Path(settings.data_dir)
_USERS_DIR = Path(settings.users_dir)
_LEGACY_LLM = _DATA / "llm_settings.json"
_LEGACY_XFYUN = _DATA / "xfyun_settings.json"
_LEGACY_GLOSSARY = _DATA / "glossary.txt"
_LEGACY_UI = _DATA / "ui_settings.json"
_INDEX_HTML = ROOT / "index.html"
# 上传的原始音视频统一留档在这里(处理完不删除,方便复用/核对;打包分享时剔除)
_UPLOAD_DIR = _DATA / "uploads"


def _user_dir(user_id: str) -> Path:
    """某个账号的私有配置目录:data/users/<id>/。"""
    return _USERS_DIR / user_id


def _llm_path(user_id: str) -> Path:
    return _user_dir(user_id) / "llm_settings.json"


def _xfyun_path(user_id: str) -> Path:
    return _user_dir(user_id) / "xfyun_settings.json"


def _ui_path(user_id: str) -> Path:
    return _user_dir(user_id) / "ui_settings.json"


def _glossary_path(user_id: str) -> Path:
    return _user_dir(user_id) / "glossary.txt"


def _migrate_legacy_settings(user_id: str) -> None:
    """把登录功能上线前全局保存的 key / 术语表复制给首个管理员,避免「升级后 key 没了」。"""
    import shutil

    for src, dst in (
        (_LEGACY_LLM, _llm_path(user_id)),
        (_LEGACY_XFYUN, _xfyun_path(user_id)),
        (_LEGACY_UI, _ui_path(user_id)),
        (_LEGACY_GLOSSARY, _glossary_path(user_id)),
    ):
        if src.exists() and not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)

_UNSAFE_NAME = re.compile(r'[\\/:*?"<>|\r\n\t]+')


def _upload_target(source_name: str, suffix: str) -> Path:
    """给上传文件生成一个可读的留档路径: 时间戳_会议名.ext(重名自动加序号)。"""
    _UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    safe = _UNSAFE_NAME.sub("_", source_name or "audio").strip(" ._") or "audio"
    safe = safe[:40]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    target = _UPLOAD_DIR / f"{stamp}_{safe}{suffix}"
    n = 2
    while target.exists():
        target = _UPLOAD_DIR / f"{stamp}_{safe}_{n}{suffix}"
        n += 1
    return target

_LLM_PRESETS: dict[str, tuple[str, str]] = {
    "DeepSeek · deepseek-chat": ("https://api.deepseek.com", "deepseek-chat"),
    "DeepSeek · deepseek-reasoner": ("https://api.deepseek.com", "deepseek-reasoner"),
    "OpenAI · gpt-4o-mini": ("https://api.openai.com/v1", "gpt-4o-mini"),
    "OpenAI · gpt-4o": ("https://api.openai.com/v1", "gpt-4o"),
    "通义千问 · qwen-plus": ("https://dashscope.aliyuncs.com/compatible-mode/v1", "qwen-plus"),
    "通义千问 · qwen-turbo": ("https://dashscope.aliyuncs.com/compatible-mode/v1", "qwen-turbo"),
}
_LLM_OPTIONS = list(_LLM_PRESETS.keys()) + ["自定义…"]
_ASR_MODELS = ["tiny", "base", "small", "medium"]

_DEFAULT_UI = {
    "language": "",
    "translate_target": "",
    "asr_model": "small",
    "enable_correct": True,
    "enable_diarize": True,
    "enable_speaker_llm": True,
    "num_speakers": 0,
    "live_lang": "zh-CN",
    "live_target": "en",
}


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""


def _ollama_available() -> bool:
    """探测本机 Ollama 是否在运行且已拉取配置的模型。"""
    import urllib.request

    root = settings.ollama_base_url.rstrip("/").rsplit("/v1", 1)[0]
    try:
        with urllib.request.urlopen(root + "/api/tags", timeout=2) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return False
    names = {m.get("name", "").split(":")[0] for m in data.get("models", [])}
    return settings.ollama_model.split(":")[0] in names


def _llm_config(user_id: str) -> tuple[str, str, str, bool]:
    """读取用户在自己设置页填的 LLM 配置;未填 key 时回退本机 Ollama。"""
    saved = _read_json(_llm_path(user_id))
    api_key = saved.get("api_key") or ""
    provider = saved.get("provider") or "DeepSeek · deepseek-chat"
    if provider == "自定义…":
        base_url = saved.get("base_url") or settings.llm_base_url
        model = saved.get("model") or settings.llm_model
    else:
        base_url, model = _LLM_PRESETS.get(provider, _LLM_PRESETS["DeepSeek · deepseek-chat"])
    using_ollama = not api_key
    if using_ollama:
        base_url, model, api_key = settings.ollama_base_url, settings.ollama_model, "ollama"
    return api_key, base_url, model, using_ollama


# ============ 登录 / 会话 ============
def require_user(request: Request) -> dict:
    """受保护接口的依赖:从 cookie 取会话,无效直接 401。"""
    token = request.cookies.get(SESSION_COOKIE)
    user = get_user_by_token(token) if token else None
    if not user:
        raise HTTPException(status_code=401, detail="请先登录")
    return user


def _set_session_cookie(response: Response, token: str, user: dict) -> None:
    """游客用会话 cookie(关浏览器即失效);正式账号 7 天免登录。"""
    kwargs: dict[str, Any] = {
        "key": SESSION_COOKIE,
        "value": token,
        "httponly": True,
        "samesite": "lax",
        "path": "/",
    }
    if not user.get("is_guest"):
        kwargs["max_age"] = SESSION_DAYS * 86400
    response.set_cookie(**kwargs)


class Credentials(BaseModel):
    username: str = ""
    password: str = ""


@router.get("/api/auth/me")
async def api_auth_me(request: Request) -> dict:
    """前端启动时调用:未登录时靠 setup_required 决定显示「首建管理员」还是「登录」。"""
    token = request.cookies.get(SESSION_COOKIE)
    user = get_user_by_token(token) if token else None
    return {"setup_required": count_users() == 0, "user": user}


@router.post("/api/auth/register")
async def api_register(body: Credentials, response: Response) -> dict:
    try:
        user = create_user(body.username, body.password)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if user.get("is_admin"):
        # 首个管理员接管旧记录与旧设置文件
        backfill_orphan_runs(user["id"])
        _migrate_legacy_settings(user["id"])
    token = create_session(user["id"])
    _set_session_cookie(response, token, user)
    return {"user": user}


@router.post("/api/auth/login")
async def api_login(body: Credentials, response: Response) -> dict:
    user = authenticate(body.username, body.password)
    if not user:
        raise HTTPException(status_code=401, detail="用户名或密码错误")
    token = create_session(user["id"])
    _set_session_cookie(response, token, user)
    return {"user": user}


@router.post("/api/auth/guest")
async def api_guest(response: Response) -> dict:
    user = create_guest()
    token = create_session(user["id"])
    _set_session_cookie(response, token, user)
    return {"user": user}


@router.post("/api/auth/upgrade")
async def api_upgrade(
    body: Credentials,
    request: Request,
    response: Response,
    user: dict = Depends(require_user),
) -> dict:
    """游客原地升级为正式账号,数据(记录 / 设置)不迁移、不丢失。"""
    if not user.get("is_guest"):
        raise HTTPException(status_code=400, detail="当前账号不是游客")
    try:
        upgraded = upgrade_guest(user["id"], body.username, body.password)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    # 换发一个正式账号的会话 cookie(带 7 天有效期)
    delete_session(request.cookies.get(SESSION_COOKIE) or "")
    token = create_session(upgraded["id"])
    _set_session_cookie(response, token, upgraded)
    return {"user": upgraded}


@router.post("/api/auth/logout")
async def api_logout(request: Request, response: Response) -> dict:
    token = request.cookies.get(SESSION_COOKIE)
    user = get_user_by_token(token) if token else None
    if token:
        delete_session(token)
    if user and user.get("is_guest"):
        # 游客登出即清空其记录与设置
        purge_user(user["id"])
    response.delete_cookie(SESSION_COOKIE, path="/")
    return {"ok": True}


# ============ 任务(后台线程 + 轮询) ============
_JOBS: dict[str, dict] = {}
_JOBS_LOCK = threading.Lock()

_STEPS = [
    ("asr", "语音转写"),
    ("diar", "说话人识别"),
    ("correct", "同音字纠错"),
    ("llm", "纪要 + 翻译"),
]


def _new_job(user_id: str) -> str:
    job_id = uuid.uuid4().hex[:12]
    with _JOBS_LOCK:
        _JOBS[job_id] = {
            "state": "running",
            "steps": {k: {"state": "idle", "detail": "等待中…"} for k, _ in _STEPS},
            "error": "",
            "run_id": "",
            "user_id": user_id,
            "started_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "warnings": [],
        }
    return job_id


def _set_step(job_id: str, key: str, state: str, detail: str = "") -> None:
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        if job is not None:
            job["steps"][key] = {"state": state, "detail": detail}


def _job_done(job_id: str, run_id: str) -> None:
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        if job is not None:
            job["state"] = "done"
            job["run_id"] = run_id


def _job_failed(job_id: str, message: str) -> None:
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        if job is not None:
            job["state"] = "failed"
            job["error"] = message


def _run_pipeline(
    job_id: str,
    *,
    user_id: str,
    audio_path: str | None,
    text: str,
    source_name: str,
    language: str | None,
    translate_target: str,
    glossary: str,
    enable_correct: bool,
    enable_diarize: bool,
    num_speakers: int,
    enable_speaker_llm: bool,
    asr_model: str,
    instruction: str = "",
) -> None:
    """后台线程里跑完整流水线;结果写入历史记录并回填任务状态。"""
    api_key, base_url, model, using_ollama = _llm_config(user_id)
    clear_fallback_notices()
    source_segments: list[dict] | None = None
    source_text = ""
    duration: float | None = None
    try:
        if audio_path:
            _set_step(job_id, "asr", "running", "加载模型…")
            duration_est = probe_duration(audio_path)
            spans = None
            diar_thread: threading.Thread | None = None
            diar_error: list[str] = []

            def _do_diar() -> None:
                nonlocal spans
                try:
                    spans = diarize(
                        audio_path,
                        num_speakers=num_speakers or None,
                        duration=duration_est,
                    )
                except Exception as exc:  # pragma: no cover - 依赖/模型相关
                    diar_error.append(str(exc))

            if enable_diarize:
                _set_step(job_id, "diar", "running", "与转写同时进行…")
                diar_thread = threading.Thread(target=_do_diar, daemon=True)
                diar_thread.start()
            else:
                _set_step(job_id, "diar", "done", "已关闭")

            segment_count = {"n": 0}

            def _on_segment(seg) -> None:  # noqa: ANN001 - callback from asr
                segment_count["n"] += 1
                if segment_count["n"] % 5 == 0:
                    _set_step(job_id, "asr", "running", f"已转写 {segment_count['n']} 段…")

            prompt = build_initial_prompt(language, glossary)
            transcript = transcribe_file(
                audio_path,
                language=language,
                initial_prompt=prompt,
                model_size=asr_model,
                on_segment=_on_segment,
            )
            duration = transcript.duration
            source_segments = segments_to_simplified(list(transcript.to_dict()["segments"]))
            source_text = to_simplified(transcript.text)
            _set_step(
                job_id,
                "asr",
                "done",
                f"{len(source_segments)} 段 · {transcript.elapsed_seconds:.0f}s",
            )

            if diar_thread is not None:
                diar_thread.join()
                if diar_error:
                    _set_step(job_id, "diar", "failed", f"识别失败,已跳过:{diar_error[0][:40]}")
                elif spans:
                    source_segments = assign_speakers(source_segments, spans)
                    n_people = len({s["speaker"] for s in source_segments if s.get("speaker")})
                    _set_step(job_id, "diar", "done", f"{n_people} 位讲话人")
                else:
                    _set_step(job_id, "diar", "done", "未检测到讲话人")
        else:
            source_text = to_simplified(text)
            _set_step(job_id, "asr", "done", "文本输入,跳过")
            if enable_speaker_llm:
                _set_step(job_id, "diar", "running", "LLM 说话人分析…")
                try:
                    source_segments = annotate_speakers(
                        source_text, model=model, api_key=api_key, base_url=base_url
                    )
                    n_people = len({s.get("speaker") for s in source_segments if s.get("speaker")})
                    _set_step(job_id, "diar", "done", f"{n_people} 位讲话人")
                except Exception as exc:  # pragma: no cover - 网络/结构异常
                    source_segments = [{"start": 0.0, "end": 0.0, "text": source_text}]
                    _set_step(job_id, "diar", "failed", f"分析失败,按单人处理:{str(exc)[:30]}")
            else:
                source_segments = [{"start": 0.0, "end": 0.0, "text": source_text}]
                _set_step(job_id, "diar", "done", "已关闭")

        # ---- 同音字纠错 ----
        if enable_correct:
            _set_step(job_id, "correct", "running", "同音字 + 术语统一…")
            corrected = segments_to_simplified(
                correct_transcript(
                    source_segments,
                    glossary=glossary,
                    model=model,
                    api_key=api_key,
                    base_url=base_url,
                )
            )
            corrected_text = to_simplified("".join(s.get("text", "") for s in corrected))
            _set_step(job_id, "correct", "done", f"{len(corrected)} 段已校对")
        else:
            corrected = source_segments
            corrected_text = source_text
            _set_step(job_id, "correct", "done", "已关闭")

        # ---- 纪要 + 翻译(并行) ----
        has_speaker = any(s.get("speaker") for s in corrected)
        summary_input = (
            render_dialogue(corrected, with_timestamp=False) if has_speaker else corrected_text
        )
        target = translate_target or None
        translated_lang = langs.label_of(target) if target else ""
        _set_step(
            job_id,
            "llm",
            "running",
            "纪要 + 翻译并行中…" if target else "生成结构化纪要…",
        )

        from concurrent.futures import ThreadPoolExecutor

        summary: dict = {}
        brief = ""
        translation: Any = None
        pool = ThreadPoolExecutor(max_workers=3)
        try:
            sum_future = pool.submit(
                summarize,
                summary_input,
                instruction=instruction or None,
                model=model,
                api_key=api_key,
                base_url=base_url,
            )
            brief_future = pool.submit(
                brief_summary,
                summary_input,
                instruction=instruction or None,
                model=model,
                api_key=api_key,
                base_url=base_url,
            )
            if target:
                if has_speaker:
                    tr_future = pool.submit(
                        translate_segments,
                        corrected,
                        target=target,
                        glossary=glossary,
                        model=model,
                        api_key=api_key,
                        base_url=base_url,
                    )
                else:
                    tr_future = pool.submit(
                        translate,
                        corrected_text,
                        target=target,
                        glossary=glossary,
                        model=model,
                        api_key=api_key,
                        base_url=base_url,
                    )
            else:
                tr_future = None
            summary = deep_to_simplified(sum_future.result())
            try:
                brief = to_simplified(brief_future.result() or "")
            except Exception:  # pragma: no cover - 精简总结失败不阻断主流程
                brief = ""
            translation = tr_future.result() if tr_future else None
        finally:
            pool.shutdown(wait=False)

        translated_summary = None
        translated_text = None
        if translation is not None:
            _set_step(job_id, "llm", "running", f"翻译纪要成{translated_lang}…")
            translated_summary = translate_summary(
                summary,
                target=target,
                glossary=glossary,
                model=model,
                api_key=api_key,
                base_url=base_url,
            )
            if has_speaker:
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
                translated_text = translation
        _set_step(
            job_id,
            "llm",
            "done",
            f"{len(summary.get('topics', []) or [])} 议题 · "
            f"{len(summary.get('action_items', []) or [])} 待办"
            + (f" · 已译{translated_lang}" if translated_lang else ""),
        )

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
            "has_timestamps": audio_path is not None,
            "fallback_local": fallback_local,
        }
        meta = save_run(source_name, result, user_id=user_id)
        with _JOBS_LOCK:
            job = _JOBS.get(job_id)
            if job is not None:
                job["warnings"] = ["云端不可达,已改用本机 Ollama"] if fallback_local else []
        _job_done(job_id, meta["id"])
    except Exception as exc:  # pragma: no cover - 模型/网络相关
        _job_failed(job_id, f"{type(exc).__name__}: {exc}")
    # 注意:上传的原始音视频处理完不再删除,留档在 data/uploads/ 下,可手动清理。


# ============ 页面 ============
@router.get("/", include_in_schema=False)
async def index_page() -> FileResponse:
    if not _INDEX_HTML.exists():
        raise HTTPException(status_code=500, detail="index.html 不存在")
    return FileResponse(_INDEX_HTML, headers={"Cache-Control": "no-store"})


@router.get("/ui", include_in_schema=False)
async def ui_alias() -> FileResponse:
    return await index_page()


@router.get("/s/{token}", include_in_schema=False)
async def shared_page(token: str) -> HTMLResponse:
    """只读分享页:凭 token 打开,不需要登录;取消分享 / 记录进回收站后立即失效。"""
    data = get_shared_run(token)
    if not data:
        return HTMLResponse(
            "<!DOCTYPE html><html lang=\"zh-CN\"><head><meta charset=\"UTF-8\">"
            "<title>分享链接无效</title></head>"
            "<body style=\"font-family:system-ui,'PingFang SC','Microsoft YaHei',sans-serif;"
            "display:flex;min-height:100vh;align-items:center;justify-content:center;"
            "background:#f6f7fb;color:#475569\">"
            "<div style=\"text-align:center\"><p style=\"font-size:18px;font-weight:600\">分享链接无效或已取消</p>"
            "<p style=\"font-size:13px;color:#94a3b8\">请联系分享者重新生成链接。</p></div>"
            "</body></html>",
            status_code=404,
            headers={"Cache-Control": "no-store"},
        )
    return HTMLResponse(
        render_share_page(data["result"], data["meta"]),
        headers={"Cache-Control": "no-store", "X-Robots-Tag": "noindex, nofollow"},
    )


# ============ 元信息 / 设置 ============
@router.get("/api/meta")
async def api_meta(request: Request, user: dict = Depends(require_user)) -> dict:
    user_id = user["id"]
    saved_llm = _read_json(_llm_path(user_id))
    saved_xf = _read_json(_xfyun_path(user_id))
    ui = {**_DEFAULT_UI, **_read_json(_ui_path(user_id))}
    scheme = "wss" if request.url.scheme == "https" else "ws"
    host = request.headers.get("host") or "127.0.0.1:8000"
    return {
        "llm": {
            "api_key": saved_llm.get("api_key", ""),
            "provider": saved_llm.get("provider", "DeepSeek · deepseek-chat"),
            "base_url": saved_llm.get("base_url", ""),
            "model": saved_llm.get("model", ""),
            "presets": {k: list(v) for k, v in _LLM_PRESETS.items()},
            "options": _LLM_OPTIONS,
        },
        "xfyun": {"appid": saved_xf.get("appid", ""), "apikey": saved_xf.get("apikey", "")},
        "glossary": _read_text(_glossary_path(user_id)),
        "ui": ui,
        "langs": [{"code": code, "label": v["label"]} for code, v in langs.LANGUAGES.items()],
        "asr_models": _ASR_MODELS,
        "ollama": {
            "model": settings.ollama_model,
            "base_url": settings.ollama_base_url,
            "available": _ollama_available(),
        },
        "rtasr_ws_url": f"{scheme}://{host}/rtasr",
        "runs": list_runs(user_id),
    }


class SettingsBody(BaseModel):
    llm: dict | None = None
    xfyun: dict | None = None
    glossary: str | None = None
    ui: dict | None = None


@router.post("/api/settings")
async def api_save_settings(
    body: SettingsBody, user: dict = Depends(require_user)
) -> dict:
    user_id = user["id"]
    if body.llm is not None:
        provider = body.llm.get("provider") or "DeepSeek · deepseek-chat"
        if provider == "自定义…":
            base_url = body.llm.get("base_url") or settings.llm_base_url
            model = body.llm.get("model") or settings.llm_model
        else:
            base_url, model = _LLM_PRESETS.get(provider, _LLM_PRESETS["DeepSeek · deepseek-chat"])
        _write_json(
            _llm_path(user_id),
            {
                "api_key": body.llm.get("api_key") or "",
                "provider": provider,
                "base_url": base_url,
                "model": model,
            },
        )
    if body.xfyun is not None:
        _write_json(
            _xfyun_path(user_id),
            {
                "appid": body.xfyun.get("appid") or "",
                "apikey": body.xfyun.get("apikey") or "",
            },
        )
    if body.glossary is not None:
        glossary_path = _glossary_path(user_id)
        glossary_path.parent.mkdir(parents=True, exist_ok=True)
        glossary_path.write_text(body.glossary, encoding="utf-8")
    if body.ui is not None:
        _write_json(_ui_path(user_id), {**_DEFAULT_UI, **body.ui})
    return {"ok": True}


# ============ 历史记录 ============
class FavoriteBody(BaseModel):
    favorite: bool = True


@router.get("/api/runs")
async def api_runs(user: dict = Depends(require_user)) -> list[dict]:
    return list_runs(user["id"])


@router.get("/api/runs/trash")
async def api_trash_runs(user: dict = Depends(require_user)) -> dict:
    return {
        "runs": list_deleted_runs(user["id"]),
        "retention_days": TRASH_RETENTION_DAYS,
    }


@router.get("/api/runs/favorites")
async def api_favorite_runs(user: dict = Depends(require_user)) -> dict:
    return {"runs": list_favorite_runs(user["id"])}


@router.get("/api/runs/shares")
async def api_shared_runs(
    request: Request, user: dict = Depends(require_user)
) -> dict:
    """「我的分享」列表;share_url 用当前访问地址拼,换端口 / 局域网访问都能对。"""
    origin = str(request.base_url).rstrip("/")
    runs = list_shared_runs(user["id"])
    for run in runs:
        run["share_url"] = f"{origin}/s/{run['share_token']}" if run.get("share_token") else ""
    return {"runs": runs}


@router.get("/api/runs/{run_id}")
async def api_run(run_id: str, user: dict = Depends(require_user)) -> dict:
    user_id = user["id"]
    result = get_run(run_id, user_id)
    if not result:
        raise HTTPException(status_code=404, detail="记录不存在或文件已丢失")
    meta = next((r for r in list_runs(user_id) if r["id"] == run_id), None)
    return {"id": run_id, "meta": meta, "result": result}


@router.delete("/api/runs/{run_id}")
async def api_delete_run(run_id: str, user: dict = Depends(require_user)) -> dict:
    if not delete_run(run_id, user["id"]):
        raise HTTPException(status_code=404, detail="记录不存在")
    return {"ok": True, "trashed": True}


@router.post("/api/runs/{run_id}/restore")
async def api_restore_run(run_id: str, user: dict = Depends(require_user)) -> dict:
    if not restore_run(run_id, user["id"]):
        raise HTTPException(status_code=404, detail="记录不存在")
    return {"ok": True, "restored": True}


@router.delete("/api/runs/{run_id}/purge")
async def api_purge_run(run_id: str, user: dict = Depends(require_user)) -> dict:
    if not purge_run(run_id, user["id"]):
        raise HTTPException(status_code=404, detail="记录不存在或不在回收站")
    return {"ok": True, "purged": True}


@router.post("/api/runs/{run_id}/favorite")
async def api_favorite_run(
    run_id: str, body: FavoriteBody, user: dict = Depends(require_user)
) -> dict:
    state = set_favorite(run_id, body.favorite, user["id"])
    if state is None:
        raise HTTPException(status_code=404, detail="记录不存在或已在回收站")
    return {"ok": True, "favorite": state}


class ShareBody(BaseModel):
    shared: bool = True


@router.post("/api/runs/{run_id}/share")
async def api_share_run(
    run_id: str,
    body: ShareBody,
    request: Request,
    user: dict = Depends(require_user),
) -> dict:
    """生成 / 取消分享链接。重复调用返回同一个链接,已分享的记录不会换 token。"""
    if not body.shared:
        if not cancel_share(run_id, user["id"]):
            raise HTTPException(status_code=404, detail="记录不存在或未分享")
        return {"ok": True, "shared": False}
    token = create_share(run_id, user["id"])
    if not token:
        raise HTTPException(status_code=404, detail="记录不存在或已在回收站")
    url = f"{str(request.base_url).rstrip('/')}/s/{token}"
    return {"ok": True, "shared": True, "token": token, "url": url}


@router.get("/api/runs/{run_id}/export")
async def api_export(
    run_id: str, fmt: str = "md", user: dict = Depends(require_user)
) -> Response:
    result = get_run(run_id, user["id"])
    if not result:
        raise HTTPException(status_code=404, detail="记录不存在")
    fmt = fmt.lower()
    body, mime = render_svc.export_text(result, fmt)
    suffix = {
        "md": "md",
        "bilingual": "bilingual.md",
        "json": "summary.json",
        "srt": "srt",
        "txt": "txt",
        "translated": "translated.md",
    }.get(fmt, "md")
    name = (result.get("source") or "meeting").replace("/", "_").replace("\\", "_")
    return Response(
        content=body,
        media_type=mime + "; charset=utf-8",
        headers={
            "Content-Disposition": f"attachment; filename*=UTF-8''{_url_quote(name)}.{suffix}"
        },
    )


def _url_quote(text: str) -> str:
    from urllib.parse import quote

    return quote(text)


class TranslateBody(BaseModel):
    scope: str = "text"
    target: str = "en"


@router.post("/api/runs/{run_id}/translate")
async def api_translate_run(
    run_id: str, body: TranslateBody, user: dict = Depends(require_user)
) -> dict:
    result = get_run(run_id, user["id"])
    if not result:
        raise HTTPException(status_code=404, detail="记录不存在")
    api_key, base_url, model, _ = _llm_config(user["id"])
    glossary = _read_text(_glossary_path(user["id"]))
    corrected = result.get("corrected_segments") or []
    has_speaker = any(s.get("speaker") for s in corrected)
    try:
        if body.scope == "summary":
            summary = result.get("summary") or {}
            if not summary:
                raise HTTPException(status_code=400, detail="没有可翻译的纪要")
            value = translate_summary(
                summary,
                target=body.target,
                glossary=glossary,
                model=model,
                api_key=api_key,
                base_url=base_url,
            )
            dialogue = (
                render_dialogue(corrected, with_timestamp=False)
                if has_speaker
                else (result.get("corrected_text") or result.get("transcript") or "")
            )
            text = render_svc.to_markdown(
                value,
                dialogue,
                is_dialogue=has_speaker,
                lang=langs.markdown_lang(body.target),
            )
            return {"type": "summary", "value": value, "markdown": text}
        source = (
            render_dialogue(corrected, with_timestamp=False)
            if has_speaker
            else (result.get("corrected_text") or result.get("transcript") or "")
        )
        if not source.strip():
            raise HTTPException(status_code=400, detail="没有可翻译的文本")
        value = translate(
            source,
            target=body.target,
            glossary=glossary,
            model=model,
            api_key=api_key,
            base_url=base_url,
        )
        return {"type": "text", "value": value}
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - 网络/配额相关
        raise HTTPException(status_code=502, detail=f"翻译失败:{exc}") from exc


class AskBody(BaseModel):
    question: str
    history: list[dict] = []


@router.post("/api/runs/{run_id}/ask")
async def api_ask(
    run_id: str, body: AskBody, user: dict = Depends(require_user)
) -> dict:
    result = get_run(run_id, user["id"])
    if not result:
        raise HTTPException(status_code=404, detail="记录不存在")
    question = (body.question or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="问题不能为空")
    api_key, base_url, model, _ = _llm_config(user["id"])
    clear_fallback_notices()
    try:
        answer = ask_meeting(
            render_svc.qa_context(result),
            question,
            history=body.history[-10:],
            model=model,
            api_key=api_key,
            base_url=base_url,
        )
    except Exception as exc:  # pragma: no cover - 网络/配额相关
        raise HTTPException(status_code=502, detail=f"问答失败:{exc}") from exc
    return {"answer": answer, "fallback_local": bool(pop_fallback_notices())}


class LiveTranslateBody(BaseModel):
    text: str
    target: str = "en"
    glossary: str = ""


@router.post("/api/live/translate")
async def api_live_translate(
    body: LiveTranslateBody, user: dict = Depends(require_user)
) -> dict:
    text = (body.text or "").strip()
    if not text:
        return {"text": ""}
    api_key, base_url, model, _ = _llm_config(user["id"])
    glossary = body.glossary or _read_text(_glossary_path(user["id"]))
    try:
        value = translate(
            text,
            target=body.target,
            glossary=glossary,
            model=model,
            api_key=api_key,
            base_url=base_url,
        )
    except Exception as exc:  # pragma: no cover - 网络/配额相关
        raise HTTPException(status_code=502, detail=f"翻译失败:{exc}") from exc
    return {"text": value}


# ============ 生成任务 ============
@router.post("/api/jobs")
async def api_create_job(
    file: UploadFile | None = File(None),
    text: str = Form(""),
    title: str = Form(""),
    source: str = Form("上传音频"),
    language: str = Form(""),
    translate_target: str = Form(""),
    glossary: str = Form(""),
    enable_correct: bool = Form(True),
    enable_diarize: bool = Form(True),
    num_speakers: int = Form(0),
    enable_speaker_llm: bool = Form(True),
    asr_model: str = Form("small"),
    prompt: str = Form(""),
    user: dict = Depends(require_user),
) -> dict:
    user_id = user["id"]
    text = (text or "").strip()
    audio_path: str | None = None
    source_name = (title or "").strip()
    if file is not None and file.filename:
        data = await file.read()
        if not data:
            raise HTTPException(status_code=400, detail="上传的文件为空")
        suffix = Path(file.filename).suffix or ".wav"
        if not source_name:
            source_name = Path(file.filename).stem
        target = _upload_target(source_name, suffix)
        target.write_bytes(data)
        audio_path = str(target)
    elif not text:
        raise HTTPException(status_code=400, detail="请上传音频或粘贴会议文本")
    else:
        source_name = source_name or ("实时字幕" if source == "实时字幕" else "文本输入")

    if not _read_json(_llm_path(user_id)).get("api_key") and not _ollama_available():
        # 不阻断任务(有可能稍后才启动 Ollama),但在任务里会报错;这里提前给出明确提示
        pass

    job_id = _new_job(user_id)
    thread = threading.Thread(
        target=_run_pipeline,
        kwargs={
            "job_id": job_id,
            "user_id": user_id,
            "audio_path": audio_path,
            "text": text,
            "source_name": source_name,
            "language": language or None,
            "translate_target": (translate_target or "").strip(),
            "glossary": glossary or _read_text(_glossary_path(user_id)),
            "enable_correct": bool(enable_correct),
            "enable_diarize": bool(enable_diarize),
            "num_speakers": max(0, int(num_speakers or 0)),
            "enable_speaker_llm": bool(enable_speaker_llm),
            "asr_model": asr_model if asr_model in _ASR_MODELS else "small",
            "instruction": (prompt or "").strip()[:2000],
        },
        daemon=True,
    )
    thread.start()
    return {"job_id": job_id}


@router.get("/api/jobs/{job_id}")
async def api_job(job_id: str, user: dict = Depends(require_user)) -> dict:
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        # 任务只对提交它的账号可见;别人的 job_id 一律当不存在
        if job is None or job.get("user_id") != user["id"]:
            raise HTTPException(status_code=404, detail="任务不存在")
        return json.loads(json.dumps(job, ensure_ascii=False))


@router.post("/api/glossary/extract")
async def api_glossary_extract(
    file: UploadFile = File(...), user: dict = Depends(require_user)
) -> dict:
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="文件为空")
    try:
        content = extract_text(file.filename or "upload.txt", data)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"读取文件失败:{exc}") from exc
    if not content.strip():
        raise HTTPException(status_code=400, detail="文件里没有读到文字")
    api_key, base_url, model, _ = _llm_config(user["id"])
    try:
        terms = extract_terms(content, model=model, api_key=api_key, base_url=base_url)
    except Exception as exc:  # pragma: no cover - 网络/配额相关
        raise HTTPException(status_code=502, detail=f"术语提取失败:{exc}") from exc
    existing = _read_text(_glossary_path(user["id"]))
    seen = set()
    out: list[str] = []
    for item in [*existing.replace("，", ",").split(","), *(terms or [])]:
        term = str(item).strip()
        if term and term.lower() not in seen:
            seen.add(term.lower())
            out.append(term)
    merged = ", ".join(out)
    return {"terms": terms or [], "glossary": merged}
