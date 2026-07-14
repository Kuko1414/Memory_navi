#!/usr/bin/env python3
"""补全模式 CLI 薄包装（逻辑已封装进 agent_core/modes/completion.py::CompletionMode）。

验证：Qwen 能否消费 Claude 整理后的词典式地图(sub_areas + objects.abs_pose)，导航到一个
【地图里缺失/误标】的目标物体处，观测并补全它。角色分工：几何归代码、语义归 Qwen，绝不把答案坐标喂 Qwen。

运行：conda run -n vllm python complete_target.py
     （需全栈 Webots+rosbridge+MCP+vLLM 8B；只用本地 Qwen，不调云）
退出码沿用：0 SUCCESS / 1 partial / 2 setup-fail。
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
REPORT_DIR = os.path.join(REPO, "Report")
for p in (HERE, REPORT_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from agent_core import config
from agent_core.executor import Executor
from agent_core.memory.fs_memory import FsMemory
from agent_core.modes.base import RunContext
from agent_core.modes.completion import CompletionMode

AREA = "explore_room"
TARGET = "红柜"
TARGET_ALIASES = ["红色柜子", "red_cabinet", "cabinet", "柜子"]
CABINET_KEYS = ["柜", "cabinet", "红"]


def main() -> int:
    mem = FsMemory(config.MEMORY_ROOT, config.ENV_NAME)
    try:
        ex = Executor()
    except Exception as e:  # noqa: BLE001
        print(f"❌ 无法建立 Executor: {e}")
        return 2
    print(f"[vLLM] model = {ex.model}")
    try:
        ctx = RunContext(
            ex=ex, mem=mem, area=AREA, task=f"补全{TARGET}", report_dir=REPORT_DIR,
            params={"target": TARGET, "target_aliases": TARGET_ALIASES,
                    "cabinet_keys": CABINET_KEYS},
        )
        res = CompletionMode().run(ctx)
    finally:
        ex.close()
    print(f"\n[结果] finish={res.finish} status={res.status} {res.duration_s}s")
    print(f"  summary: {res.summary}")
    return res.exit_code


if __name__ == "__main__":
    sys.exit(main())
