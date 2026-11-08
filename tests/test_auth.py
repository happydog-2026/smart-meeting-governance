"""账号 / 会话 / 数据隔离的回归测试。"""

import datetime
import sqlite3

import pytest

from app.services import auth, history


@pytest.fixture
def tmp_db(tmp_path, monkeypatch):
    """把 auth 与 history 的数据库 / 目录都指到临时目录,不碰真实 data/。"""
    db = tmp_path / "history.db"
    hist = tmp_path / "history"
    users = tmp_path / "users"
    hist.mkdir()
    monkeypatch.setattr(auth, "DB_PATH", db)
    monkeypatch.setattr(auth, "USERS_DIR", users)
    monkeypatch.setattr(history, "DB_PATH", db)
    monkeypatch.setattr(history, "HISTORY_DIR", hist)
    return tmp_path


def _result(source="demo"):
    return {
        "source": source,
        "duration": 1.0,
        "summary": {"title": source, "topics": [], "action_items": []},
        "corrected_segments": [],
    }


def test_password_roundtrip():
    stored = auth.hash_password("secret1")
    assert stored.startswith("scrypt$")
    assert auth.verify_password("secret1", stored)
    assert not auth.verify_password("wrong", stored)
    assert not auth.verify_password("secret1", "garbage")


def test_first_user_is_admin_and_duplicate_rejected(tmp_db):
    alice = auth.create_user("alice", "pass1")
    bob = auth.create_user("bob", "pass2")
    assert alice["is_admin"] is True
    assert bob["is_admin"] is False
    assert alice["is_guest"] is False
    with pytest.raises(ValueError, match="已存在"):
        auth.create_user("alice", "pass3")
    assert auth.count_users() == 2
    assert auth.authenticate("alice", "pass1")["id"] == alice["id"]
    assert auth.authenticate("alice", "bad") is None


def test_session_expiry(tmp_db):
    user = auth.create_user("alice", "pass1")
    token = auth.create_session(user["id"])
    assert auth.get_user_by_token(token)["id"] == user["id"]

    past = (datetime.datetime.now() - datetime.timedelta(days=1)).strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    conn = sqlite3.connect(auth.DB_PATH)
    conn.execute("UPDATE sessions SET expires_at = ? WHERE token = ?", (past, token))
    conn.commit()
    conn.close()

    assert auth.get_user_by_token(token) is None
    assert auth.get_user_by_token("no-such-token") is None


def test_history_isolation_between_users(tmp_db):
    alice = auth.create_user("alice", "pass1")
    bob = auth.create_user("bob", "pass2")
    history.save_run("alice-meeting", _result(), user_id=alice["id"])
    history.save_run("bob-meeting", _result(), user_id=bob["id"])

    assert [r["source_name"] for r in history.list_runs(alice["id"])] == ["alice-meeting"]
    assert [r["source_name"] for r in history.list_runs(bob["id"])] == ["bob-meeting"]

    # 旧调用方(不传 user_id)仍然能拿到全部记录
    assert len(history.list_runs()) == 2

    # 别人的记录读不到 / 删不掉
    run = history.list_runs(alice["id"])[0]
    assert history.get_run(run["id"], bob["id"]) is None
    assert history.get_run(run["id"], alice["id"]) is not None
    assert history.delete_run(run["id"], bob["id"]) is False
    assert history.delete_run(run["id"], alice["id"]) is True


def test_guest_upgrade_keeps_id_and_data(tmp_db):
    guest = auth.create_guest()
    assert guest["is_guest"] is True
    history.save_run("guest-meeting", _result(), user_id=guest["id"])
    user_dir = auth.USERS_DIR / guest["id"]
    user_dir.mkdir(parents=True)
    (user_dir / "glossary.txt").write_text("DeepSeek", encoding="utf-8")

    upgraded = auth.upgrade_guest(guest["id"], "carol", "pass1")

    assert upgraded["id"] == guest["id"]  # 同一个 id,数据不用迁移
    assert upgraded["is_guest"] is False
    assert upgraded["username"] == "carol"
    assert [r["source_name"] for r in history.list_runs(guest["id"])] == ["guest-meeting"]
    assert (user_dir / "glossary.txt").read_text(encoding="utf-8") == "DeepSeek"
    assert auth.authenticate("carol", "pass1")["id"] == guest["id"]


def test_guest_purge_clears_data(tmp_db):
    guest = auth.create_guest()
    history.save_run("guest-meeting", _result(), user_id=guest["id"])
    user_dir = auth.USERS_DIR / guest["id"]
    user_dir.mkdir(parents=True)
    (user_dir / "glossary.txt").write_text("x", encoding="utf-8")

    auth.purge_user(guest["id"])

    assert history.list_runs(guest["id"]) == []
    assert auth.get_user(guest["id"]) is None
    assert not user_dir.exists()


def test_backfill_orphan_runs(tmp_db):
    admin = auth.create_user("admin", "pass1")
    history.save_run("old-meeting", _result())  # 登录功能上线前的记录,无归属

    assert auth.backfill_orphan_runs(admin["id"]) == 1
    assert [r["source_name"] for r in history.list_runs(admin["id"])] == ["old-meeting"]
