"""HTTP 层:登录保护、按用户隔离、游客升级 / 清空的端到端测试。

依赖 httpx(FastAPI TestClient 需要);未安装时整文件跳过,不影响其它测试。
"""

import pytest

pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from app import webui  # noqa: E402
from app.main import app  # noqa: E402
from app.services import auth, history  # noqa: E402


def _result(source="demo"):
    return {
        "source": source,
        "duration": 1.0,
        "summary": {"title": source, "topics": [], "action_items": []},
        "corrected_segments": [],
    }


@pytest.fixture
def client(tmp_path, monkeypatch):
    """把数据库与所有设置路径指到临时目录,避免污染真实 data/。"""
    db = tmp_path / "history.db"
    hist = tmp_path / "history"
    hist.mkdir()
    users = tmp_path / "users"
    monkeypatch.setattr(auth, "DB_PATH", db)
    monkeypatch.setattr(auth, "USERS_DIR", users)
    monkeypatch.setattr(history, "DB_PATH", db)
    monkeypatch.setattr(history, "HISTORY_DIR", hist)
    monkeypatch.setattr(webui, "_DATA", tmp_path)
    monkeypatch.setattr(webui, "_USERS_DIR", users)
    monkeypatch.setattr(webui, "_LEGACY_LLM", tmp_path / "llm_settings.json")
    monkeypatch.setattr(webui, "_LEGACY_XFYUN", tmp_path / "xfyun_settings.json")
    monkeypatch.setattr(webui, "_LEGACY_GLOSSARY", tmp_path / "glossary.txt")
    monkeypatch.setattr(webui, "_LEGACY_UI", tmp_path / "ui_settings.json")
    with TestClient(app) as c:
        yield c


def test_protected_endpoints_require_login(client):
    assert client.get("/api/runs").status_code == 401
    assert client.get("/api/meta").status_code == 401
    me = client.get("/api/auth/me").json()
    assert me == {"setup_required": True, "user": None}


def test_register_then_isolated_histories(client):
    alice = client.post(
        "/api/auth/register", json={"username": "alice", "password": "pass1"}
    ).json()["user"]
    assert alice["is_admin"] is True
    history.save_run("alice-run", _result(), user_id=alice["id"])
    assert [r["source_name"] for r in client.get("/api/runs").json()] == ["alice-run"]

    assert client.post("/api/auth/logout").status_code == 200
    assert client.get("/api/runs").status_code == 401

    bob = client.post(
        "/api/auth/register", json={"username": "bob", "password": "pass2"}
    ).json()["user"]
    assert bob["is_admin"] is False
    assert client.get("/api/runs").json() == []
    # 别人的记录按 id 也读不到
    run_id = history.list_runs(alice["id"])[0]["id"]
    assert client.get(f"/api/runs/{run_id}").status_code == 404


def test_guest_uses_session_cookie_and_upgrade_keeps_data(client):
    resp = client.post("/api/auth/guest")
    assert resp.status_code == 200
    guest = resp.json()["user"]
    assert guest["is_guest"] is True
    # 游客 cookie 不带 Max-Age,关闭浏览器即失效
    assert "max-age" not in resp.headers.get("set-cookie", "").lower()

    history.save_run("guest-run", _result(), user_id=guest["id"])
    upgraded = client.post(
        "/api/auth/upgrade", json={"username": "carol", "password": "pass1"}
    ).json()["user"]
    assert upgraded["id"] == guest["id"]
    assert upgraded["is_guest"] is False
    assert [r["source_name"] for r in client.get("/api/runs").json()] == ["guest-run"]

    client.post("/api/auth/logout")
    assert (
        client.post(
            "/api/auth/login", json={"username": "carol", "password": "pass1"}
        ).status_code
        == 200
    )
    # 升级后的正式账号退出再登录,数据仍在
    assert len(client.get("/api/runs").json()) == 1


def test_guest_logout_purges_their_data(client):
    guest = client.post("/api/auth/guest").json()["user"]
    history.save_run("guest-run", _result(), user_id=guest["id"])
    assert client.get("/api/runs").json()

    client.post("/api/auth/logout")

    assert history.list_runs(guest["id"]) == []
    assert auth.get_user(guest["id"]) is None


def test_task_only_visible_to_owner(client):
    alice = client.post(
        "/api/auth/register", json={"username": "alice", "password": "pass1"}
    ).json()["user"]
    job_id = webui._new_job(alice["id"])
    assert client.get(f"/api/jobs/{job_id}").json()["state"] == "running"

    client.post("/api/auth/logout")
    client.post("/api/auth/register", json={"username": "bob", "password": "pass2"})
    assert client.get(f"/api/jobs/{job_id}").status_code == 404
