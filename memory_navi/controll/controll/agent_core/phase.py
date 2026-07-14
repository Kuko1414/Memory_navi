"""确定性阶段检测（supervisor 的唯一决策 seam）。

三阶段本质是【纯记忆状态查表】，无可 embedding 的语义量——故用确定性代码，不引词向量/第三模型
（对齐 proposal“supervisor 走确定性代码”）：
  area.json 不存在                         -> 'explore'
  area 存在但任务目标物体在图中不可解析     -> 'completion'
  area 存在且任务目标物体已在图中           -> 'execution'

模糊的“任务文本→记忆物体”解析复用现有 `_norm_name` 别名/子串匹配（非 embedding）；
权威语义解释交给执行模式内部的 Claude plan_route。

轻依赖：只引 fs_memory 的字符串归一化，保证可被纯单测导入而不拉起 MCP/openai 栈。
detect_phase 可整体替换（若某天真要 A/B 词向量变体，只换这一个函数）。
"""
from agent_core.memory.fs_memory import _norm_name

Phase = str  # "explore" | "completion" | "execution"


def target_resolvable(rec: dict, task: str, *, targets=None) -> bool:
    """任务目标是否已在地图中可解析（确定性字符串/别名匹配，非 embedding）。

    - targets（执行实验显式给定的目标列表 [{name,...}]）：任一 name 在图中匹配即为真。
    - 否则退化到 task 文本：对每个物体的 name+aliases 做归一化子串双向包含匹配
      （镜像 execution._via_candidates 的匹配口径）。
    """
    objs = rec.get("objects", []) if isinstance(rec, dict) else []

    def _hit(query: str) -> bool:
        nq = _norm_name(query)
        if not nq:
            return False
        for o in objs:
            for n in [o.get("name")] + (o.get("aliases") or []):
                if n and (_norm_name(n) in nq or nq in _norm_name(n)):
                    return True
        return False

    if targets:
        return any(_hit(t.get("name", "")) for t in targets if isinstance(t, dict))
    return _hit(task or "")


def detect_phase(mem, area: str, task: str, *, targets=None) -> Phase:
    """纯记忆状态检测。mem 需有 load_area(area)->dict|None。"""
    rec = mem.load_area(area)
    if rec is None:
        return "explore"
    if target_resolvable(rec, task, targets=targets):
        return "execution"
    return "completion"
