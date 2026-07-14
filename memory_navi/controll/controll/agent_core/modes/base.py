"""Mode 抽象基类 + 运行上下文 + 同构回报 + 记忆权限代理。

设计要点（对齐 proposal 分工与"相互独立、不数据污染"）：
- 每个 Mode 声明 memory_access ∈ {none,read,write,rewrite}；base.run 据此把 FsMemory 包一层代理，
  read/none 物理上写不了、write 只放行 upsert_object/add_edge——非污染是【结构性保证】而非约定。
- ModeResult 三模式同构：status/finish/summary/exit_code/duration_s/metrics/artifacts，
  供 supervisor 统一回报（是否 finish + Qwen 总结 + 耗时 + 简要指标）。
- LLM 只出现在子类 do_run 内部；base 不碰模型（summary 兜底用 Qwen，但失败即模板兜底，不阻断）。

轻依赖：仅在 TYPE_CHECKING 下引用 Executor，保证 base 可被纯单测导入而不拉起 MCP/openai 栈。
"""
from __future__ import annotations

import abc
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

from agent_core import config
from agent_core.memory.fs_memory import FsMemory

if TYPE_CHECKING:                      # 仅类型标注，不在运行期导入重依赖
    from agent_core.executor import Executor


# ---- 非污染契约：四个声明级别 ----
MEMORY_ACCESS = ("none", "read", "write", "rewrite")
#   none    : 代理拦截一切读写   —— 纯运动/感知（本轮无此模式）
#   read    : load_* 放行；一切写抛 PermissionError —— execution（只读地图，从不写）
#   write   : read + 仅放行 {upsert_object, add_edge}（追加补物体） —— completion
#   rewrite : 完整写面（含 upsert_area / apply_curation） —— explore 合法删+重建 area.json

_WRITE_METHODS = {"upsert_area", "upsert_object", "add_edge", "apply_curation"}
_READ_METHODS = {"load_area", "list_areas", "load_topology", "topology_path", "_area_path"}
_WRITE_SCOPED_ALLOW = {"upsert_object", "add_edge"}   # write 级别放行的写方法


class ReadOnlyMemory:
    """透明只读代理：读转发给 FsMemory，任何写方法抛 PermissionError。

    allow_read=False 时连读也拦（memory_access='none'）。用 __getattr__ 转发，
    故 FsMemory 未来新增读方法自动可用；写方法白/黑名单显式列出。
    """

    def __init__(self, inner: FsMemory, *, allow_read: bool):
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "_allow_read", allow_read)

    def __getattr__(self, name):
        if name in _WRITE_METHODS:
            raise PermissionError(f"memory_access 禁止写方法 '{name}'")
        if not object.__getattribute__(self, "_allow_read") and name in _READ_METHODS:
            raise PermissionError(f"memory_access='none' 禁止读方法 '{name}'")
        return getattr(object.__getattribute__(self, "_inner"), name)


class WriteScopedMemory:
    """限定写面代理（memory_access='write'）：读全放行，只放行 {upsert_object, add_edge}，

    upsert_area / apply_curation 等"重写/整理"级写抛 PermissionError——让 completion 的
    "只补一个物体"成为结构性保证，杜绝误调整理/覆盖。
    """

    def __init__(self, inner: FsMemory):
        object.__setattr__(self, "_inner", inner)

    def __getattr__(self, name):
        if name in _WRITE_METHODS and name not in _WRITE_SCOPED_ALLOW:
            raise PermissionError(f"memory_access='write' 只放行 {_WRITE_SCOPED_ALLOW}，禁止 '{name}'")
        return getattr(object.__getattribute__(self, "_inner"), name)


@dataclass
class RunContext:
    """Mode 运行所需的一切，由 Supervisor 注入（取代模块级 AREA/TARGET/REPORT_DIR 全局）。"""

    ex: "Executor"
    mem: FsMemory                        # 原始 handle；base.run 会按 memory_access 包代理后再交给 do_run
    director: Any = None                 # make_director(...) 结果；uses_cloud=False 时为 None
    env: str = config.ENV_NAME
    area: str = "explore_room"           # 原模块级 AREA
    task: str = ""                       # 自由文本任务
    params: dict = field(default_factory=dict)   # 每模式扩展（TARGET / targets / manner ...）
    report_dir: str = ""                 # 原模块级 REPORT_DIR
    rules_state: dict = field(default_factory=dict)   # rule-0/1/2 envelope（Phase 4 钩子）


@dataclass
class ModeResult:
    """统一结束契约——三模式同构（finish/report 契约）。"""

    mode: str                            # 元数据 name
    status: str                          # 模式原生判定 "SUCCESS"|"NOT_ARRIVED"|"NO_MAP"|...
    finish: bool                         # supervisor 级：任务阶段是否完成且满足
    summary: str = ""                    # Qwen 生成的自然语言总结（base 在缺省时补齐）
    exit_code: int = 0                   # 保留旧脚本退出码 0 ok / 1 partial / 2 setup-fail
    duration_s: float = 0.0              # 墙钟耗时（base.run 填）
    metrics: dict = field(default_factory=dict)     # 数值：dist_m/steps/sweeps/targets...
    artifacts: dict = field(default_factory=dict)   # 路径：run json / review dir / area path
    error: Optional[str] = None


class Mode(abc.ABC):
    """技能模式基类：声明式元数据 + run(守卫/计时/保证 summary) → 子类 do_run。"""

    # ---- 类级声明式元数据（子类覆盖）----
    name: str = "base"
    skill: str = ""                      # 一行人读描述
    allowlist: Optional[set] = None      # None -> Executor 默认 config.ACTION_ALLOWLIST
    memory_access: str = "read"          # MEMORY_ACCESS 之一
    uses_cloud: bool = False             # True -> supervisor 建 director 挂到 ctx

    def run(self, ctx: RunContext) -> ModeResult:
        """强制非污染 + 计时 + 保证 summary，然后调 do_run。"""
        if self.memory_access not in MEMORY_ACCESS:
            raise ValueError(f"未知 memory_access={self.memory_access!r}")
        guarded = self._guard_memory(ctx.mem)
        wrapped = RunContext(
            ex=ctx.ex, mem=guarded, director=ctx.director, env=ctx.env,
            area=ctx.area, task=ctx.task, params=ctx.params,
            report_dir=ctx.report_dir, rules_state=ctx.rules_state,
        )
        t0 = time.time()
        try:
            res = self.do_run(wrapped)
        except PermissionError as e:      # 模式违反自己声明的记忆权限
            res = ModeResult(self.name, "MEMORY_VIOLATION", False,
                             summary=str(e), exit_code=2, error=str(e))
        res.duration_s = round(time.time() - t0, 1)
        if not res.summary:
            res.summary = self._author_summary(wrapped, res)
        return res

    def _guard_memory(self, mem: FsMemory):
        """按 memory_access 包记忆代理。rewrite/write 需真写，返回原 handle 或限定写面。"""
        if self.memory_access == "rewrite":
            return mem                     # 完整写面（explore 删+重建）
        if self.memory_access == "write":
            return WriteScopedMemory(mem)  # 只放行补物体
        return ReadOnlyMemory(mem, allow_read=(self.memory_access == "read"))

    @abc.abstractmethod
    def do_run(self, ctx: RunContext) -> ModeResult:
        """子类实现：搬入原 main() 体，收到已按 memory_access 包过代理的 ctx。"""

    def _author_summary(self, ctx: RunContext, res: ModeResult) -> str:
        """模式没给 summary 时兜底：优先让 Qwen 用一句话总结结果，失败即模板兜底。"""
        template = (f"[{self.name}] {res.status}；finish={res.finish}；"
                    f"metrics={res.metrics}")
        ex = ctx.ex
        if ex is None or getattr(ex, "client", None) is None:
            return template
        try:
            prompt = (f"用一句不超过40字的中文总结机器人本次『{self.name}』任务结果："
                      f"判定={res.status}，是否完成={res.finish}，指标={res.metrics}。只输出这句话。")
            r = ex.client.chat.completions.create(
                model=ex.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2, max_tokens=80, stream=False)
            txt = (r.choices[0].message.content or "").strip()
            return txt or template
        except Exception:                  # noqa: BLE001 —— 总结兜底绝不阻断任务
            return template
