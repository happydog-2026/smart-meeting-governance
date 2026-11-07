"""历史记录:每次运行的结果存档与回看。

存储分两层,互为冗余:
- SQLite(data/history.db):轻量元数据,用于列表快速展示,不加载完整结果。
- JSON(data/history/<id>.json):完整结果,点开某条时才读。

只依赖标准库 sqlite3,不引入新依赖。
"""

from __future__ import annotations

import json
import secrets
import sqlite3
import uuid
from datetime import datetime, timedelta
from pathlib import Path

from app.config import settings

HISTORY_DIR: Path = settings.data_dir / "history"
DB_PATH: Path = settings.data_dir / "history.db"

#: 回收站保留天数:软删除超过这个天数会被彻底清理
TRASH_RETENTION_DAYS = 30

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
    json_path TEXT NOT NULL,
    is_deleted INTEGER NOT NULL DEFAULT 0,
    deleted_at TEXT,
    is_favorite INTEGER NOT NULL DEFAULT 0,
    favorited_at TEXT,
    share_token TEXT,
    shared_at TEXT
)
"""


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _migrate(conn: sqlite3.Connection) -> None:
    """给旧库补上软删除 / 收藏 / 分享字段(幂等)。"""
    cols = {row["name"] for row in conn.execute("PRAGMA table_info(history)")}
    if "is_deleted" not in cols:
        conn.execute("ALTER TABLE history ADD COLUMN is_deleted INTEGER NOT NULL DEFAULT 0")
    if "deleted_at" not in cols:
        conn.execute("ALTER TABLE history ADD COLUMN deleted_at TEXT")
    if "is_favorite" not in cols:
        conn.execute("ALTER TABLE history ADD COLUMN is_favorite INTEGER NOT NULL DEFAULT 0")
    if "favorited_at" not in cols:
        conn.execute("ALTER TABLE history ADD COLUMN favorited_at TEXT")
    if "share_token" not in cols:
        conn.execute("ALTER TABLE history ADD COLUMN share_token TEXT")
    if "shared_at" not in cols:
        conn.execute("ALTER TABLE history ADD COLUMN shared_at TEXT")


def _db() -> sqlite3.Connection:
    """打开数据库并确保表存在(历史目录一并创建)。"""
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute(_SCHEMA)
    _migrate(conn)
    conn.commit()
    return conn


def _row_to_dict(row: sqlite3.Row, keys: list[str]) -> dict:
    return {k: row[k] for k in keys}


_COLS = [
    "id",
    "source_name",
    "created_at",
    "duration",
    "n_speakers",
    "n_topics",
    "n_actions",
    "title",
    "is_favorite",
    "share_token",
    "shared_at",
    "json_path",
]


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
    """返回未删除的历史记录元数据,按时间倒序(最新在前)。"""
    conn = _db()
    try:
        rows = conn.execute(
            f"SELECT {', '.join(_COLS)} FROM history WHERE is_deleted = 0 ORDER BY created_at DESC"
        ).fetchall()
    finally:
        conn.close()
    return [_row_to_dict(r, _COLS) for r in rows]


def list_deleted_runs() -> list[dict]:
    """返回回收站里的记录,按删除时间倒序;顺带清理超过保留期的记录。"""
    purge_expired()
    cols = [*_COLS, "deleted_at"]
    conn = _db()
    try:
        rows = conn.execute(
            f"SELECT {', '.join(cols)} FROM history WHERE is_deleted = 1 ORDER BY deleted_at DESC"
        ).fetchall()
    finally:
        conn.close()
    return [_row_to_dict(r, cols) for r in rows]


def list_favorite_runs() -> list[dict]:
    """返回已收藏且未删除的记录,按收藏时间倒序(最新收藏在前)。"""
    cols = [*_COLS, "favorited_at"]
    conn = _db()
    try:
        rows = conn.execute(
            f"SELECT {', '.join(cols)} FROM history "
            "WHERE is_deleted = 0 AND is_favorite = 1 "
            "ORDER BY favorited_at DESC, created_at DESC"
        ).fetchall()
    finally:
        conn.close()
    return [_row_to_dict(r, cols) for r in rows]


def set_favorite(run_id: str, favorite: bool) -> bool | None:
    """收藏 / 取消收藏一条记录。

    只允许操作未删除的记录;记录不存在或已在回收站时返回 None,
    否则返回设置后的收藏状态。
    """
    conn = _db()
    try:
        row = conn.execute(
            "SELECT id FROM history WHERE id = ? AND is_deleted = 0", (run_id,)
        ).fetchone()
        if row is None:
            return None
        conn.execute(
            "UPDATE history SET is_favorite = ?, favorited_at = ? WHERE id = ?",
            (1 if favorite else 0, _now() if favorite else None, run_id),
        )
        conn.commit()
    finally:
        conn.close()
    return bool(favorite)


def create_share(run_id: str) -> str | None:
    """为一条未删除的记录创建分享 token;已分享则沿用原 token(链接不变)。

    只允许分享未删除的记录;记录不存在或已在回收站时返回 None。
    """
    conn = _db()
    try:
        row = conn.execute(
            "SELECT share_token FROM history WHERE id = ? AND is_deleted = 0", (run_id,)
        ).fetchone()
        if row is None:
            return None
        token = row["share_token"]
        if token:
            return token
        token = secrets.token_urlsafe(9)
        while conn.execute("SELECT 1 FROM history WHERE share_token = ?", (token,)).fetchone():
            token = secrets.token_urlsafe(9)
        conn.execute(
            "UPDATE history SET share_token = ?, shared_at = ? WHERE id = ?",
            (token, _now(), run_id),
        )
        conn.commit()
        return token
    finally:
        conn.close()


def cancel_share(run_id: str) -> bool:
    """取消分享,原链接立即失效;记录不存在或本就没分享返回 False。"""
    conn = _db()
    try:
        row = conn.execute(
            "SELECT id FROM history WHERE id = ? AND share_token IS NOT NULL", (run_id,)
        ).fetchone()
        if row is None:
            return False
        conn.execute(
            "UPDATE history SET share_token = NULL, shared_at = NULL WHERE id = ?", (run_id,)
        )
        conn.commit()
    finally:
        conn.close()
    return True


def list_shared_runs() -> list[dict]:
    """返回正在分享且未删除的记录,按分享时间倒序。"""
    conn = _db()
    try:
        rows = conn.execute(
            f"SELECT {', '.join(_COLS)} FROM history "
            "WHERE is_deleted = 0 AND share_token IS NOT NULL "
            "ORDER BY shared_at DESC, created_at DESC"
        ).fetchall()
    finally:
        conn.close()
    return [_row_to_dict(r, _COLS) for r in rows]


def get_shared_run(token: str) -> dict | None:
    """按分享 token 读取记录(只读页用);已取消分享 / 已删除 / 文件丢失返回 None。"""
    if not token:
        return None
    conn = _db()
    try:
        row = conn.execute(
            f"SELECT {', '.join(_COLS)}, json_path FROM history "
            "WHERE share_token = ? AND is_deleted = 0",
            (token,),
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        return None
    path = Path(row["json_path"])
    if not path.exists():
        return None
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return {"meta": _row_to_dict(row, _COLS), "result": result}


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
    """软删除:把记录移入回收站,保留 30 天后由 purge_expired 彻底清理。

    返回是否找到并可删除(记录不存在返回 False)。
    """
    conn = _db()
    try:
        row = conn.execute("SELECT id FROM history WHERE id = ?", (run_id,)).fetchone()
        if row is None:
            return False
        conn.execute(
            "UPDATE history SET is_deleted = 1, deleted_at = ? WHERE id = ?",
            (_now(), run_id),
        )
        conn.commit()
    finally:
        conn.close()
    return True


def restore_run(run_id: str) -> bool:
    """从回收站恢复一条记录;记录不存在返回 False。"""
    conn = _db()
    try:
        row = conn.execute("SELECT id FROM history WHERE id = ?", (run_id,)).fetchone()
        if row is None:
            return False
        conn.execute(
            "UPDATE history SET is_deleted = 0, deleted_at = NULL WHERE id = ?",
            (run_id,),
        )
        conn.commit()
    finally:
        conn.close()
    return True


def purge_run(run_id: str) -> bool:
    """彻底删除回收站里的一条记录(数据库行 + JSON 文件)。

    只允许彻底删除已软删除的记录;不在回收站或不存在返回 False。
    """
    conn = _db()
    try:
        row = conn.execute(
            "SELECT json_path FROM history WHERE id = ? AND is_deleted = 1", (run_id,)
        ).fetchone()
        if row is None:
            return False
        conn.execute("DELETE FROM history WHERE id = ?", (run_id,))
        conn.commit()
    finally:
        conn.close()
    Path(row["json_path"]).unlink(missing_ok=True)
    return True


def purge_expired(days: int = TRASH_RETENTION_DAYS) -> int:
    """彻底删除回收站里超过保留期的记录,返回清理条数。"""
    cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")
    rows: list[sqlite3.Row] = []
    conn = _db()
    try:
        rows = conn.execute(
            "SELECT id, json_path FROM history "
            "WHERE is_deleted = 1 AND deleted_at IS NOT NULL AND deleted_at < ?",
            (cutoff,),
        ).fetchall()
        if rows:
            conn.executemany("DELETE FROM history WHERE id = ?", [(r["id"],) for r in rows])
            conn.commit()
    finally:
        conn.close()
    for row in rows:
        Path(row["json_path"]).unlink(missing_ok=True)
    return len(rows)
