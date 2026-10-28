"""讯飞「实时语音转写」(rtasr)WebSocket 客户端辅助。

只放纯函数:签名、URL 拼接、结果解析。真正的网络连接在 app/main.py 的中继里做,
便于单元测试且与传输层解耦。

协议要点(讯飞开放平台·实时语音转写标准版):
- 握手:wss://rtasr.xfyun.cn/v1/ws?appid={appid}&ts={ts}&signa={signa}&lang={lang}
- 签名:signa = base64(HmacSHA1(key=apiKey, msg=MD5(appid + ts) 的十六进制字符串))
- 音频:16k / 16bit / 单声道 PCM,建议每 40ms 发 1280 字节(640 采样点)
- 结束:客户端发一个 binary message,内容为 {"end": true}
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import time
import urllib.parse

ENDPOINT = "wss://rtasr.xfyun.cn/v1/ws"


def build_signa(appid: str, ts: str, api_key: str) -> str:
    """生成握手签名。已用官方文档示例值验证:
    appid=595f23df, ts=1512041814, apiKey=d9f4aa7ea6d94faca62cd88a28fd5234
    => signa=IrrzsJeOFk1NGfJHW6SkHUoN9CU=
    """
    base_string = appid + ts
    md5_hex = hashlib.md5(base_string.encode("utf-8")).hexdigest()
    digest = hmac.new(api_key.encode("utf-8"), md5_hex.encode("utf-8"), hashlib.sha1).digest()
    return base64.b64encode(digest).decode("ascii")


def build_url(appid: str, api_key: str, *, lang: str = "cn", ts: str | None = None, **extra: str) -> str:
    """拼接握手 URL。extra 可传 pd / roleType / engLangType 等可选参数。"""
    ts = ts or str(int(time.time()))
    query = {"appid": appid, "ts": ts, "signa": build_signa(appid, ts, api_key)}
    if lang:
        query["lang"] = lang
    query.update({k: v for k, v in extra.items() if v not in (None, "")})
    return f"{ENDPOINT}?{urllib.parse.urlencode(query)}"


def parse_result(data: dict) -> dict:
    """从讯飞返回的 data(已 json.loads)提取文本与状态。

    返回 {"text", "final", "seg_id"}:
    - text:   本段识别文本(把 rt[].ws[].cw[].w 逐词拼接)
    - final:  True 表示最终结果(type=0),False 表示中间结果(type=1)
    - seg_id: 结果序号(从 0 起),用于客户端去重/对齐
    """
    cn = data.get("cn") or {}
    st = cn.get("st") or {}
    rt = st.get("rt") or []

    words: list[str] = []
    for sent in rt:
        for ws in sent.get("ws", []):
            for cw in ws.get("cw", []):
                w = cw.get("w")
                if w:
                    words.append(w)

    typ = str(st.get("type", "1"))
    return {
        "text": "".join(words),
        "final": typ == "0",
        "seg_id": data.get("seg_id", 0),
    }
