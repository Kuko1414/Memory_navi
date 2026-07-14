"""确定性 Supervisor：纯记忆状态检测阶段 → 映射到 Mode → 统一回报。决策环内无 AI。

对齐 CLAUDE.md 6 条确定性规则的优先级信封：
  rule 0 安全（lidar<0.15 或 sensors_ok==False → EmergencyStop）—— v1 文档化钩子 _safety_precheck
  rule 1 云触发（entered_new_area / skill==inspect → DispatchCloud）—— 由 Mode 内部承担
  rule 2 retry>=3 → SafePause+Notify —— v1 钩子（rules_state）
  rule 3/4/5 explore/navigate/complete —— 即 detect_phase 的三分支（本模块的唯一决策）

确定性保证：dispatch/detect_phase 内无 chat.completions、无 director.*、无置信度阈值；
LLM 只在 Mode.do_run 内部（Qwen 选地标 / Claude 规划路线），从不参与"选哪个模式"。
"""
import json
import os

from agent_core import config
from agent_core.cloud.providers import make_director
from agent_core.executor import Executor
from agent_core.memory.fs_memory import FsMemory
from agent_core.modes.base import ModeResult, RunContext
from agent_core.modes.completion import CompletionMode
from agent_core.modes.execution import ExecutionMode
from agent_core.modes.explore import ExploreMode
from agent_core.phase import detect_phase

_MODE_FOR_PHASE = {
    "explore": ExploreMode,
    "completion": CompletionMode,
    "execution": ExecutionMode,
}


class Supervisor:
    """按阶段确定性分发 Mode，并产出三模式同构的结束回报。"""

    def __init__(self, env=config.ENV_NAME, report_dir="", *, detect=detect_phase):
        # detect 可注入 —— 换阶段检测 seam 不动 dispatch（词向量变体的挂点）。
        self.env = env
        self.report_dir = report_dir
        self._detect = detect
        self.mem = FsMemory(config.MEMORY_ROOT, env)

    def dispatch(self, area: str, task: str, *, params=None, ex=None) -> ModeResult:
        params = params or {}
        # ---- rule 0 安全信封（v1 钩子）----
        self._safety_precheck()
        # ---- rules 3/4/5：阶段→模式（唯一决策，纯函数，无 LLM）----
        phase = self._detect(self.mem, area, task, targets=params.get("targets"))
        mode = _MODE_FOR_PHASE[phase]()
        print(f"[SUPERVISOR] 阶段判定 phase={phase} → mode={mode.name}"
              f"（memory_access={mode.memory_access}, uses_cloud={mode.uses_cloud}）")

        own_ex = ex is None
        if own_ex:
            ex = Executor(allowlist=mode.allowlist)
        director = make_director(model=config.CURATE_MODEL) if mode.uses_cloud else None
        ctx = RunContext(
            ex=ex, mem=self.mem, director=director, env=self.env,
            area=area, task=task, params=params, report_dir=self.report_dir,
        )
        try:
            res = mode.run(ctx)
        finally:
            if own_ex and ex is not None:
                ex.close()
        self._report(phase, res)
        return res

    # ---- rule 0 钩子：读 safety_node 共享状态（v1 只告警，不阻断；Phase 4 补 EmergencyStop）----
    def _safety_precheck(self):
        ns = config.AGENT_NS
        shm = f"/dev/shm/agent_safety_{ns}.json"
        try:
            if os.path.exists(shm):
                with open(shm, encoding="utf-8") as f:
                    st = json.load(f)
                if st.get("tripped"):
                    print(f"[SUPERVISOR][rule0] ⚠️ 派发前 safety tripped（lidar_min={st.get('lidar_min_m')}m）")
        except Exception:  # noqa: BLE001
            pass

    def _report(self, phase, res: ModeResult):
        """三模式同构的统一回报（是否 finish + Qwen 总结 + 耗时 + 简要指标）。"""
        report = {
            "phase": phase, "mode": res.mode, "finish": res.finish,
            "status": res.status, "duration_s": res.duration_s,
            "summary": res.summary, "metrics": res.metrics,
        }
        print("\n" + "=" * 64)
        print(f"[SUPERVISOR 回报] phase={phase} mode={res.mode} finish={res.finish} "
              f"status={res.status} 耗时={res.duration_s}s")
        print(f"  Qwen 总结: {res.summary}")
        print(f"  指标: {res.metrics}")
        print("=" * 64)
        return report
