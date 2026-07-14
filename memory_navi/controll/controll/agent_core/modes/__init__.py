"""Mode 抽象层：每个技能一个自洽 Mode，声明式元数据 + 统一 run/report 契约。

- base.Mode：类级元数据(name/skill/allowlist/memory_access/uses_cloud) + run()/do_run()。
- RunContext / ModeResult：supervisor 与 Mode 之间的注入口与同构回报。
- 记忆权限代理：按 memory_access 结构性保证"非污染"（read/none 物理写不了、write 只放行补物体）。
"""
from agent_core.modes.base import (
    MEMORY_ACCESS,
    Mode,
    ModeResult,
    ReadOnlyMemory,
    RunContext,
    WriteScopedMemory,
)

__all__ = [
    "MEMORY_ACCESS",
    "Mode",
    "ModeResult",
    "ReadOnlyMemory",
    "RunContext",
    "WriteScopedMemory",
]
