#!/usr/bin/env python3
"""执行模式 / 语义路线导航 CLI 薄包装（逻辑已封装进 agent_core/modes/execution.py::ExecutionMode）。

Claude 规划语义路线(via+manner，不出坐标) + Qwen 逐段语义导航 + 底层 VFH 避障 + 末段安全靠近观测。
默认沿用原单目标语义路线示例（东侧工位区最东绿植的左边）；执行实验用多目标见 run_execution_experiment.py。

运行：conda run -n vllm python semantic_route.py（需全栈 + ANTHROPIC_AUTH_TOKEN 云网关规划）
退出码：0 SUCCESS / 1 partial / 2 setup-fail。
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
from agent_core.cloud.providers import make_director
from agent_core.executor import Executor
from agent_core.memory.fs_memory import FsMemory
from agent_core.modes.base import RunContext
from agent_core.modes.execution import ExecutionMode

AREA = "explore_room"
# 默认单目标（等价原 semantic_route 示例）：到东侧工位区最东绿植的左边。
PARAMS = {
    "target_desc": "到『东侧工位区』里最靠东的那株绿植的【左边】",
    "target_subarea": "东侧工位区",
    "target_name": "绿植",
    "target_manner": "left",
}


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
            ex=ex, mem=mem, director=make_director(model=config.CURATE_MODEL),
            area=AREA, task="语义路线导航", report_dir=REPORT_DIR, params=PARAMS,
        )
        res = ExecutionMode().run(ctx)
    finally:
        ex.close()
    print(f"\n[结果] finish={res.finish} status={res.status} {res.duration_s}s")
    print(f"  summary: {res.summary}")
    return res.exit_code


if __name__ == "__main__":
    sys.exit(main())
