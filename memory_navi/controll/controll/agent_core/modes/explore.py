"""探索模式（Mode 封装，wrap-not-rewrite）。

explore_probe.py 有 2698 行实验脚手架（占据栅格/frontier/YOLO/dual/hybrid review）——【绝不搬】。
本模式只调用其 `run_pipeline` 单一 seam，脚手架仍是 explore_probe 私有，对其它模式结构不可见
（满足"相互独立、不数据污染"）。memory_access='rewrite'（合法删+重建 area.json）。
"""
import os
import sys

from agent_core import config
from agent_core.modes.base import Mode, ModeResult, RunContext

# explore_probe 是顶层脚本（ament 测试以 `import explore_probe as ep` 依赖其顶层可导入）。
_PKG_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # .../controll
if _PKG_DIR not in sys.path:
    sys.path.insert(0, _PKG_DIR)


class ExploreMode(Mode):
    name = "explore"
    skill = "从零覆盖建图 + Qwen 本地语义标注（占据栅格/frontier/去重/整理，全本地不调云）"
    allowlist = config.ACTION_ALLOWLIST
    memory_access = "rewrite"           # 合法删+重建 area.json，勿拦
    uses_cloud = True                   # explore_probe 内部自建 director；ctx.director 不强制

    def do_run(self, ctx: RunContext) -> ModeResult:
        import explore_probe as ep      # 顶层脚本；仅调 run_pipeline seam，不碰其 2698 行内部
        rc = ep.run_pipeline(ex=ctx.ex, mem=ctx.mem, area=ctx.area, report_dir=ctx.report_dir)
        metrics = dict(ep.LAST_RUN_METRICS)
        status = "SUCCESS" if rc == 0 else "PARTIAL"
        return ModeResult(
            self.name, status, finish=(rc == 0),
            summary="",                 # base 用 Qwen 补齐
            exit_code=rc,
            metrics=metrics,
            artifacts={"run_json": ep.RUN_OUT},
        )
