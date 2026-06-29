"""接线单测：Executor 只暴露 allowlist 工具，并把 dispatch 正确路由到 ros.call。

不连真实 vLLM/MCP：注入假 ros、假 client、固定 model，并替换 run_tool_loop 捕获入参。
可用 pytest 运行，也可 `python test/test_executor.py` 直接运行。
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.join(os.path.dirname(HERE), "controll")  # .../controll/controll
if PKG not in sys.path:
    sys.path.insert(0, PKG)

from agent_core import config  # noqa: E402
from agent_core import executor as ex_mod  # noqa: E402
from agent_core.executor import Executor  # noqa: E402


class _FakeRos:
    """假 RosTools：记录 allowlist 并提供 call 占位。"""

    def __init__(self, tools):
        """保存待返回的工具列表。"""
        self._tools = tools
        self.seen_allowlist = None

    def list_openai_tools(self, allowlist=None):
        """返回固定工具列表并记录传入的 allowlist。"""
        self.seen_allowlist = allowlist
        return self._tools

    def call(self, name, args):
        """dispatch 占位，单测里不会真正执行。"""
        return None


def _make_executor(tools):
    """构造注入了假依赖的 Executor。"""
    ros = _FakeRos(tools)
    ex = Executor(ros=ros, client=object(), model="fake-model")
    return ex, ros


def test_only_allowlist_exposed():
    """Executor 把 config.ACTION_ALLOWLIST 透传给 ros，并用其返回作为 tools。"""
    tools = [{"type": "function", "function": {"name": "move"}}]
    ex, ros = _make_executor(tools)
    assert ros.seen_allowlist == config.ACTION_ALLOWLIST
    assert ex.tools == tools


def test_run_builds_messages_and_routes_dispatch():
    """run() 构造 system+user 消息，并以 ros.call 为 dispatch、传 self.tools。"""
    tools = [{"type": "function", "function": {"name": "stop"}}]
    ex, ros = _make_executor(tools)
    captured = {}

    def _fake_loop(client, model, messages, t, dispatch, **kw):
        captured.update(messages=messages, tools=t, dispatch=dispatch, model=model)
        return "LOOP_RESULT"

    orig = ex_mod.run_tool_loop
    ex_mod.run_tool_loop = _fake_loop
    try:
        out = ex.run("前进0.5米后停", system="SYS")
    finally:
        ex_mod.run_tool_loop = orig

    assert out == "LOOP_RESULT"
    assert captured["tools"] is tools
    assert captured["dispatch"] == ros.call
    assert captured["model"] == "fake-model"
    assert captured["messages"][0] == {"role": "system", "content": "SYS"}
    assert captured["messages"][1] == {"role": "user", "content": "前进0.5米后停"}


def test_close_is_safe_without_bridge():
    """注入 ros 时无 bridge，close() 不应抛异常。"""
    ex, _ = _make_executor([])
    ex.close()


if __name__ == "__main__":
    test_only_allowlist_exposed()
    test_run_builds_messages_and_routes_dispatch()
    test_close_is_safe_without_bridge()
    print("test_executor: all PASS")
