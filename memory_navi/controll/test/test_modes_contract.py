"""Mode 契约单测（纯函数、无仿真、不连 ROS/LLM）。

覆盖：记忆权限代理(read/none/write/rewrite)的放行与拦截、ModeResult schema 字段、
base.run 的守卫（read 模式尝试写 → MEMORY_VIOLATION）与计时/summary 兜底。
在 vllm 环境跑（agent_core 顶层依赖）。
"""
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.join(os.path.dirname(HERE), "controll")  # .../controll/controll
if PKG not in sys.path:
    sys.path.insert(0, PKG)

from agent_core.memory.fs_memory import FsMemory  # noqa: E402
from agent_core.modes.base import (  # noqa: E402
    MEMORY_ACCESS,
    Mode,
    ModeResult,
    ReadOnlyMemory,
    RunContext,
    WriteScopedMemory,
)


@pytest.fixture()
def mem(tmp_path):
    m = FsMemory(str(tmp_path), "sim")
    m.upsert_area("roomA", {"area": "roomA", "type": "office",
                            "objects": [{"name": "桌子", "abs_pose": {"x": 1, "y": 0, "z": 0.5}}]})
    return m


# ---- ReadOnlyMemory ----
def test_readonly_allows_read_blocks_write(mem):
    ro = ReadOnlyMemory(mem, allow_read=True)
    assert ro.load_area("roomA")["type"] == "office"           # 读放行
    assert ro.list_areas() == ["roomA"]
    for w in ("upsert_area", "upsert_object", "add_edge", "apply_curation"):
        with pytest.raises(PermissionError):
            getattr(ro, w)                                     # 取写方法即抛


def test_none_access_blocks_even_read(mem):
    none = ReadOnlyMemory(mem, allow_read=False)
    with pytest.raises(PermissionError):
        none.load_area("roomA")


# ---- WriteScopedMemory ----
def test_writescoped_allows_upsert_object_blocks_curation(mem):
    ws = WriteScopedMemory(mem)
    assert ws.load_area("roomA") is not None                   # 读放行
    assert callable(ws.upsert_object)                          # 补物体放行
    assert callable(ws.add_edge)
    for blocked in ("upsert_area", "apply_curation"):
        with pytest.raises(PermissionError):
            getattr(ws, blocked)                               # 重写/整理级拦截
    # 放行的写真的能落盘
    ws.upsert_object("roomA", {"name": "椅子", "abs_pose": {"x": 2, "y": 0, "z": 0.5}})
    names = {o["name"] for o in mem.load_area("roomA")["objects"]}
    assert {"桌子", "椅子"} <= names


# ---- ModeResult schema ----
def test_moderesult_fields():
    r = ModeResult(mode="execution", status="SUCCESS", finish=True)
    for f in ("mode", "status", "finish", "summary", "exit_code",
              "duration_s", "metrics", "artifacts", "error"):
        assert hasattr(r, f)
    assert r.summary == "" and r.exit_code == 0 and r.metrics == {}


# ---- base.run 守卫 + 计时 + summary 兜底 ----
class _ReadModeThatWrites(Mode):
    name = "bad_read"
    memory_access = "read"

    def do_run(self, ctx):
        ctx.mem.upsert_object(ctx.area, {"name": "x"})          # 违反声明 → 应被代理拦
        return ModeResult(self.name, "SHOULD_NOT_REACH", True)


class _WriteMode(Mode):
    name = "good_write"
    memory_access = "write"

    def do_run(self, ctx):
        ctx.mem.upsert_object(ctx.area, {"name": "补的物体",
                                         "abs_pose": {"x": 3, "y": 0, "z": 0.5}})
        return ModeResult(self.name, "SUCCESS", True, summary="done", metrics={"n": 1})


def _ctx(mem):
    return RunContext(ex=None, mem=mem, area="roomA")


def test_read_mode_write_is_blocked_as_violation(mem):
    res = _ReadModeThatWrites().run(_ctx(mem))
    assert res.status == "MEMORY_VIOLATION" and res.finish is False
    assert res.exit_code == 2 and res.error


def test_write_mode_runs_and_fills_duration(mem):
    res = _WriteMode().run(_ctx(mem))
    assert res.status == "SUCCESS" and res.finish is True
    assert res.summary == "done"                               # 模式给了就不覆盖
    assert isinstance(res.duration_s, float)
    assert any(o["name"] == "补的物体" for o in mem.load_area("roomA")["objects"])


def test_summary_fallback_template_when_no_client(mem):
    class _M(Mode):
        name = "m"
        memory_access = "read"

        def do_run(self, ctx):
            return ModeResult(self.name, "NOT_ARRIVED", False, metrics={"d": 1.2})

    res = _M().run(_ctx(mem))
    assert res.summary and "NOT_ARRIVED" in res.summary          # ex=None → 模板兜底


def test_memory_access_constants():
    assert MEMORY_ACCESS == ("none", "read", "write", "rewrite")
