#!/usr/bin/env python3
"""执行模式初期实验 runner：Supervisor 确定性派发 → 执行两目标顺序任务 → 采集 → 评分 → 存档。

流程：
  1. 确保理想图已 seed（break_room_ideal）；
  2. 后台线程轮询 safety_node 共享状态采集碰撞（tripped 上升沿计数）；
  3. Supervisor.dispatch(area, task, params={targets:[两绿植]}) —— 阶段应判为 execution；
  4. 汇总 run_result（逐目标表面距离/到达、碰撞、耗时、Qwen 总结、supervisor 回报）；
  5. 存 Report/exec_run_<ts>/（run_result.json + 逐目标关键帧 + 记忆快照 + scorecard.txt）；
  6. 跑 score_execution 打分并打印；不论 PASS/FAIL 都落档，供"停下评估"。

运行：conda run -n vllm python run_execution_experiment.py
     （需全栈 Webots+rosbridge+MCP+vLLM 8B + ANTHROPIC_AUTH_TOKEN；safety_node 建议同时在跑）
"""
import json
import os
import shutil
import sys
import threading
import time
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
REPORT_DIR = os.path.join(REPO, "Report")
for p in (HERE, REPORT_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from agent_core import config
from agent_core.supervisor import Supervisor
from eval import score_execution as se
from eval.ideal_map_adapter import OUT_AREA, seed

AREA = OUT_AREA
TASK = "从起始休息区出发，先到红柜后办公区的东北绿植，再到西侧办公簇的西南绿植（从A区拿东西到B区）"
TARGETS = [
    {"name": "绿植", "x": 6.23, "y": 1.42, "manner": "near", "desc": "红柜后办公区最东北那株绿植旁"},
    {"name": "绿植", "x": -5.78, "y": -2.17, "manner": "near", "desc": "西侧办公簇最西南那株绿植旁"},
]


class CollisionPoller(threading.Thread):
    """轮询 /dev/shm/agent_safety_<ns>.json，统计 tripped 上升沿 = 碰撞次数。"""

    def __init__(self, ns, hz=5.0):
        super().__init__(daemon=True)
        self.shm = f"/dev/shm/agent_safety_{ns}.json"
        self.dt = 1.0 / hz
        self._stopped = threading.Event()   # 勿命名 _stop：会覆盖 threading.Thread._stop() 方法
        self.collisions = 0
        self.events = []
        self._last = False

    def run(self):
        while not self._stopped.is_set():
            try:
                if os.path.exists(self.shm):
                    with open(self.shm, encoding="utf-8") as f:
                        st = json.load(f)
                    tr = bool(st.get("tripped"))
                    if tr and not self._last:            # 上升沿
                        self.collisions += 1
                        self.events.append({"t": round(time.time(), 2),
                                            "lidar_min_m": st.get("lidar_min_m")})
                    self._last = tr
            except Exception:  # noqa: BLE001
                pass
            time.sleep(self.dt)

    def stop(self):
        self._stopped.set()


def _ts():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def main() -> int:
    # 首跑可先单目标：EXEC_N_TARGETS=1 只跑第 1 个目标（短、好定位）；缺省跑全部。
    targets = list(TARGETS)
    n = int(os.environ.get("EXEC_N_TARGETS", "0") or 0)
    if n > 0:
        targets = targets[:n]

    out_dir = os.path.join(REPORT_DIR, f"exec_run_{_ts()}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"[实验目录] {out_dir}  (目标数={len(targets)})")

    # 1) 确保理想图存在（幂等 seed）
    try:
        seed()
        print(f"[理想图] {AREA} 已就绪")
    except Exception as e:  # noqa: BLE001
        print(f"⚠️ 理想图 seed 失败：{e}（若已存在可忽略）")

    # 2) 碰撞采集
    poller = CollisionPoller(config.AGENT_NS)
    poller.start()

    # 3) Supervisor 派发
    sup = Supervisor(report_dir=out_dir)
    t0 = time.time()
    res = None
    try:
        res = sup.dispatch(AREA, TASK, params={"targets": targets})
    except Exception as e:  # noqa: BLE001
        print(f"❌ 派发异常：{type(e).__name__}: {e}")
    finally:
        poller.stop()
        poller.join(timeout=2.0)
    dur = round(time.time() - t0, 1)

    # 4) 汇总 run_result
    per_target = (res.metrics.get("per_target") if res else []) or []
    run_result = {
        "area": AREA, "task": TASK,
        "targets": per_target,
        "collisions": poller.collisions,
        "collision_events": poller.events,
        "duration_s": dur,
        "qwen_summary": res.summary if res else "",
        "supervisor_report": {
            # phase 名与 mode 名一一对应（explore/completion/execution），用实际派发的 mode 反映真实阶段
            "phase": res.mode if res else "unknown", "mode": res.mode if res else None,
            "finish": res.finish if res else False,
            "status": res.status if res else "ERROR",
            "duration_s": res.duration_s if res else dur,
            "summary": res.summary if res else "",
            "metrics": {k: v for k, v in (res.metrics if res else {}).items() if k != "per_target"},
        },
    }
    run_path = os.path.join(out_dir, "run_result.json")
    with open(run_path, "w", encoding="utf-8") as f:
        json.dump(run_result, f, ensure_ascii=False, indent=2)
    print(f"[run_result] {run_path}")

    # 5) 记忆快照存档
    try:
        area_json = os.path.join(config.MEMORY_ROOT, config.ENV_NAME, AREA, "area.json")
        if os.path.exists(area_json):
            shutil.copyfile(area_json, os.path.join(out_dir, "ideal_area_snapshot.json"))
    except Exception:  # noqa: BLE001
        pass

    # 6) 评分 + scorecard 落档
    gt = se.load_gt()
    s = se.score(run_result, gt)
    card = se.format_scorecard(s)
    print(card)
    with open(os.path.join(out_dir, "scorecard.txt"), "w", encoding="utf-8") as f:
        f.write(card + "\n")
    with open(os.path.join(out_dir, "score.json"), "w", encoding="utf-8") as f:
        json.dump(s, f, ensure_ascii=False, indent=2)

    print(f"\n[存档完毕] {out_dir}  (PASS={s['pass']}；初期实验：不论结果先停下评估)")
    return 0 if s["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
