"""FastAPI 后端 —— 实时语音转写 WebSocket 中继(讯飞 rtasr)。

职责:把浏览器麦克风送来的 PCM 音频实时转发给讯飞,再把讯飞的转写文本回给浏览器。
密钥来源:只来自浏览器(用户在侧边栏填的自己的讯飞 key,经首条消息送上来),
服务端不内置任何讯飞 key —— 打包分发后各人用各人的 key,不消耗开发者/部署者的。

浏览器 <-> 本服务 <-> 讯飞,协议:
- 浏览器 -> 后端:首条 text 帧 {"type":"config","appid":..,"apikey":..}(填用户自己的
  讯飞 key);此后 binary 帧为 16k/16bit/单声道 PCM(每帧 1280 字节),
  text 帧 {"type": "end"} 表示录音结束。
- 后端 -> 浏览器:text 帧:
  {"type":"status","state":"started"}   握手成功
  {"type":"result","text","final","seg_id"}   转写结果
  {"type":"status","state":"ended"}     结束
  {"type":"error","code","message"}     异常
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import websockets

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.xfyun_rtasr import build_url, parse_result  # noqa: E402

logger = logging.getLogger(__name__)

app = FastAPI(title="会议智治 · 实时转写中继")

# 讯飞断连后,浏览器未及时结束会话时最长等多久(秒)
XF_OPEN_TIMEOUT = 15


@app.get("/health")
async def health() -> dict:
    return {"ok": True}


@app.websocket("/rtasr")
async def rtasr_relay(ws: WebSocket) -> None:
    await ws.accept()

    appid = ""
    api_key = ""

    # 用户自己的讯飞 key(侧边栏填写)经首条 text 帧送上来;服务端不内置 key。
    try:
        first = await asyncio.wait_for(ws.receive(), timeout=5)
    except (asyncio.TimeoutError, WebSocketDisconnect):
        first = None
    if first and first.get("text"):
        try:
            obj = json.loads(first["text"])
        except json.JSONDecodeError:
            obj = {}
        if obj.get("type") == "config":
            appid = obj.get("appid") or ""
            api_key = obj.get("apikey") or ""

    if not (appid and api_key):
        await ws.send_json({"type": "error", "code": "config", "message": "未配置讯飞实时转写 key:请在「⚙️ 设置」页「🔊 实时转写 (讯飞)」填入你的 AppID 和接口密钥"})
        await ws.close()
        return

    lang = ws.query_params.get("lang", "cn")
    url = build_url(appid, api_key, lang=lang)

    try:
        async with websockets.connect(url, open_timeout=XF_OPEN_TIMEOUT, max_size=10 * 1024 * 1024) as xf:
            # 浏览器 -> 讯飞(音频 + 结束标志)
            async def browser_to_xf() -> None:
                while True:
                    try:
                        msg = await ws.receive()
                    except WebSocketDisconnect:
                        msg = {"type": "websocket.disconnect"}
                    if msg.get("type") == "websocket.disconnect":
                        await _end_xf(xf)
                        break
                    if msg.get("bytes"):
                        await xf.send(msg["bytes"])
                        continue
                    text = msg.get("text") or ""
                    try:
                        obj = json.loads(text)
                    except json.JSONDecodeError:
                        obj = {}
                    if obj.get("type") == "end":
                        await _end_xf(xf)
                        break

            # 讯飞 -> 浏览器(状态 + 转写结果)
            async def xf_to_browser() -> None:
                async for raw in xf:
                    if isinstance(raw, (bytes, bytearray)):
                        continue  # 讯飞回包是 text(JSON),二进制忽略
                    try:
                        obj = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    action = obj.get("action")
                    if action == "started":
                        await ws.send_json({"type": "status", "state": "started", "sid": obj.get("sid")})
                    elif action == "result":
                        data_str = obj.get("data") or ""
                        try:
                            data = json.loads(data_str)
                        except json.JSONDecodeError:
                            continue
                        parsed = parse_result(data)
                        if parsed["text"]:
                            await ws.send_json({"type": "result", **parsed})
                    elif action == "error":
                        await ws.send_json({"type": "error", "code": obj.get("code"), "message": obj.get("desc")})
                        return

            task = asyncio.create_task(browser_to_xf())
            try:
                await xf_to_browser()
                try:
                    await ws.send_json({"type": "status", "state": "ended"})
                except Exception:  # pragma: no cover
                    pass
            finally:
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass
    except websockets.exceptions.InvalidHandshake as exc:
        await ws.send_json({"type": "error", "code": "handshake", "message": f"讯飞握手失败:{exc}"})
    except websockets.exceptions.ConnectionClosed as exc:
        logger.warning("讯飞连接中断:%s", exc)
        try:
            await ws.send_json({"type": "error", "code": "closed", "message": f"讯飞连接中断:{exc.code}"})
        except Exception:  # pragma: no cover
            pass
    except Exception as exc:  # pragma: no cover - 兜底
        logger.exception("实时转写中继异常")
        try:
            await ws.send_json({"type": "error", "code": "internal", "message": f"中继异常:{exc}"})
        except Exception:
            pass


async def _end_xf(xf: websockets.WebSocketClientProtocol) -> None:
    """向讯飞发送结束标志(文档要求是 binary message,内容 {"end": true})。"""
    try:
        await xf.send(json.dumps({"end": True}).encode("utf-8"))
    except Exception:  # pragma: no cover - 连接可能已关闭
        pass
