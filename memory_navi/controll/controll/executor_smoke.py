#!/usr/bin/env python3
"""执行器冒烟测试：验证 Qwen 能否经 FC+MCP 调用精选动作/感知工具。

前置：rosbridge:9090、MCP server(streamable-http):9000（已加载新工具）、Webots、vLLM 8B:8000。
运行：
  conda activate vllm
  python memory_navi/controll/controll/executor_smoke.py            # 仅感知（无移动）
  python memory_navi/controll/controll/executor_smoke.py --move     # 额外跑一个小移动
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from agent_core.executor import Executor  # noqa: E402


def _show(title, res):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)
    for step in res.tool_trace:
        flag = "ERR" if step["is_error"] else "ok"
        args = json.dumps(step["args"], ensure_ascii=False)
        print(f"  - {step['name']}({args}) -> [{flag}] {step['text'][:120]}")
    print(f"[最终回答] {res.text}")
    return {s["name"] for s in res.tool_trace}


def main():
    do_move = "--move" in sys.argv
    ex = Executor()
    print(f"[vLLM] model = {ex.model}")
    names = [t["function"]["name"] for t in ex.tools]
    print(f"[MCP] 暴露给 Qwen 的工具：{names}")
    if not names:
        print("❌ allowlist 内工具一个都没注册上——MCP server 可能没重启/没加载新工具。")
        ex.close()
        return 2

    ok = True
    try:
        called = _show(
            "感知（无移动）：查前方是否畅通 + 当前位姿",
            ex.run("先用 scan_summary 检查前方是否畅通，再用 get_pose 报告你当前的位姿，然后用中文简述结果。"),
        )
        ok = ok and ("scan_summary" in called)

        if do_move:
            called2 = _show(
                "小移动：前进 0.2 米后停",
                ex.run("前进 0.2 米，然后停下。"),
            )
            ok = ok and ("move" in called2)
    finally:
        ex.close()

    print("\n" + ("SMOKE PASS ✅" if ok else "SMOKE FAIL ❌"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
