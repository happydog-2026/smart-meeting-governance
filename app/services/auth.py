"""账号与会话:多用户登录、游客模式、数据归属。

只依赖标准库(hashlib / hmac / secrets / sqlite3),与 history.py 共用同一个
SQLite 数据库(data/history.db),各用各的表:
- users     账号(含游客)
- sessions  登录会话(随机 token -> user_id,带过期时间)

数据归属约定:
- users.id 同时是 history.user_id 与 data/users/<id>/ 目录名。
- 游客(is_guest=1)没有密码,登出 / 过期时整份数据被清空。
- 游客可「升级」为正式账号:只改 username / password_hash / is_guest,
  保持同一个 id,于是历史记录与设置文件原地保留。
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import sqlite3
import uuid
from datetime import datetime, timedelta
from pathlib import Path

from app.config import settings

DB_PATH: Path = settings.data_dir / "history.db"
USERS_DIR: Path = settings.data_dir / "users"

#: 会话有效期(天)。游客的 cookie 不带 Max-Age,关闭浏览器即失效;
#: 但服务器端会话统一按这个上限兜底过期。
SESSION_DAYS = 7

#: 会话 token 的 cookie 名(webui 设置 cookie 时也用它)
SESSION_COOKIE = "meetnote_session"

# scrypt 参数:OWASP 推荐的交互式登录档位
_SCRYPT_N = 2**14
_SCRYPT_R = 8
_SCRYPT_P = 1
_MIN_PASSWORD_LEN = 4

_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,
    username TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL DEFAULT '',
    is_admin INTEGER NOT NULL DEFAULT 0,
    is_guest INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS sessions (
    token TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id);
"""


# ============ 基础设施 ============
def _now() -> datetime:
    return datetime.now()


def _fmt(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def _db() -> sqlite3.Connection:
    """打开数据库并确保账号 / 会话两张表存在(幂等)。"""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    conn.commit()
    return conn


def _public(row: sqlite3.Row) -> dict:
    """对外暴露的账号信息(绝不带 password_hash)。"""
    return {
        "id": row["id"],
        "username": row["username"],
        "is_admin": bool(row["is_admin"]),
        "is_guest": bool(row["is_guest"]),
        "created_at": row["created_at"],
    }


# ============ 密码 ============
def hash_password(password: str) -> str:
    """返回 scrypt$<salt_hex>$<hash_hex>;盐随机,同一密码每次结果都不同。"""
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(
        (password or "").encode("utf-8"),
        salt=salt,
        n=_SCRYPT_N,
        r=_SCRYPT_R,
        p=_SCRYPT_P,
    )
    return f"scrypt${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    """常数时间校验密码;格式异常一律视为不匹配。"""
    try:
        scheme, salt_hex, digest_hex = (stored or "").split("$", 2)
        if scheme != "scrypt":
            return False
        salt = bytes.fromhex(salt_hex)
        expected = bytes.fromhex(digest_hex)
    except (ValueError, AttributeError):
        return False
    digest = hashlib.scrypt(
        (password or "").encode("utf-8"),
        salt=salt,
        n=_SCRYPT_N,
        r=_SCRYPT_R,
        p=_SCRYPT_P,
    )
    return hmac.compare_digest(digest, expected)


# ============ 账号 ============
def count_users() -> int:
    """已注册(非游客)账号数量;为 0 时前端展示「创建管理员账号」。"""
    conn = _db()
    try:
        row = conn.execute("SELECT COUNT(*) AS n FROM users WHERE is_guest = 0").fetchone()
    finally:
        conn.close()
    return int(row["n"])


def get_user(user_id: str) -> dict | None:
    if not user_id:
        return None
    conn = _db()
    try:
        row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    finally:
        conn.close()
    return _public(row) if row else None


def create_user(username: str, password: str) -> dict:
    """注册正式账号;首个注册账号自动成为管理员。"""
    username = (username or "").strip()
    if not username:
        raise ValueError("用户名不能为空")
    if len(password or "") < _MIN_PASSWORD_LEN:
        raise ValueError(f"密码至少 {_MIN_PASSWORD_LEN} 位")

    conn = _db()
    try:
        if conn.execute("SELECT 1 FROM users WHERE username = ?", (username,)).fetchone():
            raise ValueError("用户名已存在")
        # 首次注册(还没有任何正式账号)的人成为管理员,负责接管旧数据
        is_first = conn.execute(
            "SELECT COUNT(*) AS n FROM users WHERE is_guest = 0"
        ).fetchone()["n"] == 0
        user_id = uuid.uuid4().hex
        conn.execute(
            "INSERT INTO users (id, username, password_hash, is_admin, is_guest, created_at) "
            "VALUES (?, ?, ?, ?, 0, ?)",
            (user_id, username, hash_password(password), 1 if is_first else 0, _fmt(_now())),
        )
        conn.commit()
        row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    finally:
        conn.close()
    return _public(row)


def authenticate(username: str, password: str) -> dict | None:
    """校验用户名 + 密码;失败返回 None(游客账号没有密码,不能这样登录)。"""
    conn = _db()
    try:
        row = conn.execute(
            "SELECT * FROM users WHERE username = ? AND is_guest = 0",
            ((username or "").strip(),),
        ).fetchone()
    finally:
        conn.close()
    if row is None or not verify_password(password, row["password_hash"]):
        return None
    return _public(row)


# ============ 游客 ============
def create_guest() -> dict:
    """创建一个无密码的临时账号。"""
    conn = _db()
    try:
        user_id = uuid.uuid4().hex
        # username 有 UNIQUE 约束,游客也生成一个不会撞车的占位名;展示层再显示「游客」
        username = "guest_" + secrets.token_hex(4)
        conn.execute(
            "INSERT INTO users (id, username, password_hash, is_admin, is_guest, created_at) "
            "VALUES (?, ?, '', 0, 1, ?)",
            (user_id, username, _fmt(_now())),
        )
        conn.commit()
        row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    finally:
        conn.close()
    return _public(row)


def upgrade_guest(user_id: str, username: str, password: str) -> dict:
    """把游客账号原地升级为正式账号(同一个 id,数据不迁移)。"""
    username = (username or "").strip()
    if not username:
        raise ValueError("用户名不能为空")
    if len(password or "") < _MIN_PASSWORD_LEN:
        raise ValueError(f"密码至少 {_MIN_PASSWORD_LEN} 位")

    conn = _db()
    try:
        row = conn.execute(
            "SELECT * FROM users WHERE id = ? AND is_guest = 1", (user_id,)
        ).fetchone()
        if row is None:
            raise ValueError("游客账号不存在或已升级")
        if conn.execute("SELECT 1 FROM users WHERE username = ?", (username,)).fetchone():
            raise ValueError("用户名已存在")
        conn.execute(
            "UPDATE users SET username = ?, password_hash = ?, is_guest = 0 WHERE id = ?",
            (username, hash_password(password), user_id),
        )
        conn.commit()
        row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    finally:
        conn.close()
    return _public(row)


# ============ 会话 ============
def create_session(user_id: str) -> str:
    token = secrets.token_urlsafe(32)
    now = _now()
    conn = _db()
    try:
        conn.execute(
            "INSERT INTO sessions (token, user_id, created_at, expires_at) VALUES (?, ?, ?, ?)",
            (token, user_id, _fmt(now), _fmt(now + timedelta(days=SESSION_DAYS))),
        )
        conn.commit()
    finally:
        conn.close()
    return token


def get_user_by_token(token: str) -> dict | None:
    """按会话 token 查账号;过期会话顺手删除并返回 None。"""
    if not token:
        return None
    now = _fmt(_now())
    conn = _db()
    try:
        row = conn.execute(
            "SELECT u.*, s.expires_at AS session_expires_at FROM sessions s "
            "JOIN users u ON u.id = s.user_id WHERE s.token = ?",
            (token,),
        ).fetchone()
        if row is None:
            return None
        if row["session_expires_at"] < now:
            conn.execute("DELETE FROM sessions WHERE token = ?", (token,))
            conn.commit()
            return None
    finally:
        conn.close()
    return _public(row)


def delete_session(token: str) -> None:
    if not token:
        return
    conn = _db()
    try:
        conn.execute("DELETE FROM sessions WHERE token = ?", (token,))
        conn.commit()
    finally:
        conn.close()


# ============ 数据清理 / 迁移 ============
def purge_user(user_id: str) -> None:
    """彻底删除一个账号及其全部数据(游客登出 / 过期清理时用)。"""
    if not user_id:
        return
    import shutil

    conn = _db()
    try:
        try:
            rows = conn.execute(
                "SELECT json_path FROM history WHERE user_id = ?", (user_id,)
            ).fetchall()
            conn.execute("DELETE FROM history WHERE user_id = ?", (user_id,))
        except sqlite3.OperationalError:  # history 表还没建出来
            rows = []
        conn.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))
        conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
        conn.commit()
    finally:
        conn.close()

    for row in rows:
        Path(row["json_path"]).unlink(missing_ok=True)
    shutil.rmtree(USERS_DIR / user_id, ignore_errors=True)


def purge_expired_guests() -> int:
    """清理「没有任何有效会话」的游客账号及其数据,返回清理数量。"""
    now = _fmt(_now())
    conn = _db()
    try:
        rows = conn.execute(
            "SELECT u.id FROM users u WHERE u.is_guest = 1 AND NOT EXISTS ("
            "  SELECT 1 FROM sessions s WHERE s.user_id = u.id AND s.expires_at >= ?"
            ")",
            (now,),
        ).fetchall()
    finally:
        conn.close()
    for row in rows:
        purge_user(row["id"])
    return len(rows)


def backfill_orphan_runs(admin_id: str) -> int:
    """把登录功能上线前的旧记录(user_id 为空)归到首个管理员名下。"""
    conn = _db()
    try:
        try:
            cur = conn.execute(
                "UPDATE history SET user_id = ? WHERE user_id IS NULL", (admin_id,)
            )
        except sqlite3.OperationalError:  # history 表还没建
            return 0
        conn.commit()
        return cur.rowcount or 0
    finally:
        conn.close()
