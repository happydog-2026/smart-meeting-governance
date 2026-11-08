"""PyAV 版本兼容垫片的回归测试。

模拟「av 19+ 不认识 metadata_errors 参数」的情况,验证:
- 传了 metadata_errors 能自动降级重试
- 与它无关的 TypeError 不会被吞掉
"""

import av
import pytest

from app.services import av_compat


def _make_fake_open(calls: list[dict]):
    """模拟 PyAV 19+:除 mode 外只认识自己实现过的参数。"""

    def fake_open(*args, **kwargs):
        calls.append(kwargs)
        unknown = [k for k in kwargs if k not in {"mode", "metadata_errors"}]
        if unknown:
            raise TypeError(
                f"open() got an unexpected keyword argument '{unknown[0]}'"
            )
        if "metadata_errors" in kwargs:
            raise TypeError(
                "open() got an unexpected keyword argument 'metadata_errors'"
            )
        return "container"

    return fake_open


def test_patch_falls_back_when_metadata_errors_unsupported(monkeypatch):
    calls: list[dict] = []
    monkeypatch.setattr(av, "open", _make_fake_open(calls))
    monkeypatch.delattr(av, av_compat._FLAG, raising=False)

    av_compat.patch_av_metadata_errors()

    container = av.open("dummy.wav", mode="r", metadata_errors="ignore")
    assert container == "container"
    # 第一次带参数失败,第二次去掉参数重试
    assert calls == [{"mode": "r", "metadata_errors": "ignore"}, {"mode": "r"}]


def test_patch_is_idempotent_and_keeps_unrelated_errors(monkeypatch):
    calls: list[dict] = []
    monkeypatch.setattr(av, "open", _make_fake_open(calls))
    monkeypatch.delattr(av, av_compat._FLAG, raising=False)

    av_compat.patch_av_metadata_errors()
    patched_open = av.open
    av_compat.patch_av_metadata_errors()
    assert av.open is patched_open  # 不会套两层

    with pytest.raises(TypeError, match="unexpected keyword"):
        av.open("dummy.wav", definitely_not_a_kwarg=1)
