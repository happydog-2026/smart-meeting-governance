"""历史记录:每次运行的结果存档与回看。

存储分两层,互为冗余:
- SQLite(data/history.db):轻量元数据,用于列表快速展示,不加载完整结果。
- JSON(data/history/<id>.json):完整结果,点开某条时才读。

只依赖标准库 sqlite3,不引入新依赖。
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path

from app.config import settings

HISTORY_DIR: Path = settings.data_dir / "history"
DB_PATH: Path = settings.data_dir / "history.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS history (
    id TEXT PRIMARY KEY,
    source_name TEXT NOT NULL,
    created_at TEXT NOT NULL,
    duration REAL,
    n_speakers INTEGER,
    n_topics INTEGER,
    n_actions INTEGER,
    title TEXT,
    json_path TEXT NOT NULL
)
"""


def _db() -> sqlite3.Connection:
    """打开数据库并确保表存在(历史目录一并创建)。"""
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute(_SCHEMA)
    conn.commit()
    return conn


def _row_to_dict(row: sqlite3.Row, keys: list[str]) -> dict:
    return {k: row[k] for k in keys}


_COLS = ["id", "source_name", "created_at", "duration", "n_speakers", "n_topics", "n_actions", "title", "json_path"]


def save_run(source_name: str, result: dict) -> dict:
    """保存一次运行结果,返回该条记录的元数据。

    Args:
        source_name: 来源名称(文件名或"文本输入")。
        result: 完整结果 dict,至少含 summary / corrected_segments / duration 字段。
    """
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    json_path = HISTORY_DIR / f"{run_id}.json"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = result.get("summary") or {}
    corrected = result.get("corrected_segments") or []
    meta = {
        "id": run_id,
        "source_name": source_name,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "duration": result.get("duration"),
        "n_speakers": len({s.get("speaker") for s in corrected if s.get("speaker")}),
        "n_topics": len(summary.get("topics") or []),
        "n_actions": len(summary.get("action_items") or []),
        "title": summary.get("title") or "",
    }

    conn = _db()
    try:
        conn.execute(
            "INSERT INTO history (id, source_name, created_at, duration, n_speakers, n_topics, n_actions, title, json_path) "
            "VALUES (:id, :source_name, :created_at, :duration, :n_speakers, :n_topics, :n_actions, :title, :json_path)",
            {**meta, "json_path": str(json_path)},
        )
        conn.commit()
    finally:
        conn.close()
    return meta


def list_runs() -> list[dict]:
    """返回所有历史记录元数据,按时间倒序(最新在前)。"""
    conn = _db()
    try:
        rows = conn.execute(f"SELECT {', '.join(_COLS)} FROM history ORDER BY created_at DESC").fetchall()
    finally:
        conn.close()
    return [_row_to_dict(r, _COLS) for r in rows]


def get_run(run_id: str) -> dict | None:
    """读取某条记录的完整结果;记录不存在或 JSON 已丢失返回 None。"""
    conn = _db()
    try:
        row = conn.execute("SELECT json_path FROM history WHERE id = ?", (run_id,)).fetchone()
    finally:
        conn.close()
    if row is None:
        return None
    path = Path(row["json_path"])
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def delete_run(run_id: str) -> bool:
    """删除一条记录(数据库行 + JSON 文件),返回是否删除成功。"""
    conn = _db()
    try:
        row = conn.execute("SELECT json_path FROM history WHERE id = ?", (run_id,)).fetchone()
        if row is None:
            return False
        conn.execute("DELETE FROM history WHERE id = ?", (run_id,))
        conn.commit()
    finally:
        conn.close()
    Path(row["json_path"]).unlink(missing_ok=True)
    return True
